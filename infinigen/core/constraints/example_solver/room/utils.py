# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei: primary author
# - Karhan Kayan: fix constants

from collections import defaultdict

import bpy
import numpy as np
import shapely
from shapely import (
    LineString,
    MultiLineString,
    Polygon,
    remove_repeated_points,
    simplify,
)
from shapely.ops import linemerge, orient, polygonize, shared_paths, unary_union

import infinigen.core.constraints.example_solver.room.constants as constants
from infinigen.assets.utils.decorate import write_co
from infinigen.assets.utils.object import new_circle
from infinigen.assets.utils.shapes import simplify_polygon
from infinigen.core.util import blender as butil

SIMPLIFY_THRESH = 1e-6
ANGLE_SIMPLIFY_THRESH = 0.2
WELD_THRESH = 0.01


def is_valid_polygon(p):
    if isinstance(p, Polygon) and p.area > 0 and p.is_valid:
        if len(p.interiors) == 0:
            return True
    return False


def canonicalize(p):
    # 修正多边形的几何问题（例如自交或无效的多边形）
    p = p.buffer(0) # 使用 buffer(0) 修复可能的几何问题
    try:
        while True:  # 不断迭代，直到多边形满足要求
             # 简化多边形并强制为2D
            p_ = shapely.force_2d(simplify_polygon(p))
            l = len(p.boundary.coords)# 获取边界坐标的数量
            if p.area == 0:# 如果多边形面积为0，抛出异常
                raise NotImplementedError("Polygon empty.")
             # 规范多边形方向（顺时针或逆时针）
            p = orient(p_)
            # 获取边界的坐标并转换为NumPy数组
            coords = np.array(p.boundary.coords[:])
            # 将坐标舍入到单位网格（通过常量UNIT控制精度）
            rounded = np.round(coords / constants.UNIT) * constants.UNIT
            # 检查坐标是否接近舍入值，如果接近则使用舍入值
            coords = np.where(
                np.all(np.abs(coords - rounded) < 1e-3, -1)[:, np.newaxis],
                rounded,
                coords,
            )
            # 计算相邻点之间的差值向量
            diff = coords[1:] - coords[:-1]
            # 归一化差值向量，避免数值问题
            diff = diff / (np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-6)
            # 计算相邻向量之间的点积，检测chui直线性
            product = (diff[[-1] + list(range(len(diff) - 1))] * diff).sum(-1)
             # 初始化有效的索引（所有点）
            valid_indices = list(range(len(coords) - 1))
            # 找到无效的点索引（点积过小或接近1）
            invalid_indices = np.nonzero((product < -0.8) | (product > 1 - 1e-6))[
                0
            ].tolist()
            # 如果存在无效点，移除其中一个（取中间点）
            if len(invalid_indices) > 0:
                i = invalid_indices[len(invalid_indices) // 2]
                valid_indices.remove(i)
            # 根据有效点重新构造多边形
            p = shapely.Polygon(coords[valid_indices + [valid_indices[0]]])
            # 如果边界坐标数量未发生变化，结束循环
            if len(p.exterior.coords) == l:
                break
        # 检查多边形是否有效，如果无效则抛出异常
        if not is_valid_polygon(p):
            raise NotImplementedError("Invalid polygon")
        return p # 返回规范化后的多边形
    except AttributeError:
        raise NotImplementedError("Invalid multi polygon")


def unit_cast(x, unit=None):
    if unit is None:
        unit = constants.UNIT
    return int(x / unit) * unit


def abs_distance(x, y):
    z = [0] * 4
    z[0 if y[0] > x[0] else 1] = np.abs(y[0] - x[0])
    z[2 if y[1] > x[1] else 3] = np.abs(y[1] - x[1])
    return np.array(z)


def update_exterior_edges(segments, shared_edges, exterior_edges=None, i=None):
    if exterior_edges is None:
        exterior_edges = {}
    for k, s in segments.items():
        if i is None or k == i:
            l = s.boundary
            for ls in shared_edges[k].values():
                l = l.difference(ls)
            if l.length > 0:
                exterior_edges[k] = (
                    MultiLineString([l]) if isinstance(l, LineString) else l
                )
            elif k in exterior_edges:
                exterior_edges.pop(k)
    return exterior_edges


def update_shared_edges(segments, shared_edges=None, i=None):
    if shared_edges is None:
        shared_edges = defaultdict(dict)
    for k, s in segments.items():
        for l, t in segments.items():
            if k != l and (i is None or k == i or l == i):
                with np.errstate(invalid="ignore"):
                    forward, backward = shared_paths(s.boundary, t.boundary).geoms
                if forward.length > 0:
                    shared_edges[k][l] = forward
                elif backward.length > 0:
                    shared_edges[k][l] = backward
                elif l in shared_edges[k]:
                    shared_edges[k].pop(l)
    return shared_edges


def update_staircase_occupancies(
    segments, staircase, staircase_occupancies=None, i=None
):
    if staircase is None:
        return None
    if staircase_occupancies is None:
        staircase_occupancies = defaultdict(dict)
    for k, s in segments.items():
        if i is None or k == i:
            staircase_occupancies[k] = s.intersection(staircase).area / staircase.area
    return staircase_occupancies


def compute_neighbours(ses, margin):
    return list(
        l for l, se in ses.items() if any(ls.length >= margin for ls in se.geoms)
    )


def linear_extend_x(base, target, new_x):
    return target[1] + (new_x - target[0]) * (base[1] - target[1]) / (
        base[0] - target[0]
    )


def linear_extend_y(base, target, new_y):
    return target[0] + (new_y - target[1]) * (base[0] - target[0]) / (
        base[1] - target[1]
    )


def cut_polygon_by_line(polygon, *args):
    merged = linemerge([polygon.boundary, *args])
    borders = unary_union(merged)
    polygons = polygonize(borders)
    return list(polygons)


def polygon2obj(p, reversed=False):
    x, y = orient(p).exterior.xy
    obj = new_circle(vertices=len(x) - 1)
    with butil.ViewportMode(obj, "EDIT"):
        bpy.ops.mesh.edge_face_add()
        if reversed:
            bpy.ops.mesh.flip_normals()
    write_co(obj, np.stack([x[:-1], y[:-1], np.zeros_like(x[:-1])], -1))
    return obj


def buffer(p, distance):
    with np.errstate(invalid="ignore"):
        return remove_repeated_points(
            simplify(p.buffer(distance, join_style="mitre"), SIMPLIFY_THRESH)
        )
