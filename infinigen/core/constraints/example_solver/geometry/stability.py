# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

from __future__ import annotations

import logging

import bmesh
import bpy
import gin
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mathutils import Vector
from shapely import MultiPolygon, Polygon
from shapely.affinity import rotate
from shapely.geometry import MultiPolygon, Point, Polygon

from infinigen.core import tagging
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.constraint_language import util as iu
from infinigen.core.constraints.example_solver import state_def

# from infinigen.core.util import blender as butil
from infinigen.core.constraints.example_solver.geometry import planes as planes
from infinigen.core.tags import Subpart
from infinigen.core.util import blender as butil

# import fcl

logger = logging.getLogger(__name__)


def project_and_align_z_with_x(polygons, z_direction):
    """
    Rotate polygons so that the Z-direction is aligned with the X-axis in 2D.

    Parameters:
    polygons (list[Polygon]): List of Shapely Polygons representing the projected 2D polygons.
    z_direction (np.array): The 2D direction vector where the Z-axis is projected.

    Returns:
    list[Polygon]: Rotated polygons with the Z-direction aligned with the X-axis.
    """
    # Calculate the angle between the Z-direction projection and the X-axis
    angle_rad = np.arctan2(z_direction[1], z_direction[0])
    angle_deg = np.degrees(angle_rad)

    # Rotate polygons to align Z-direction with X-axis
    rotated_polygons = [
        rotate(polygon, angle_deg, origin=(0, 0), use_radians=False)
        for polygon in polygons
    ]

    return rotated_polygons


def is_vertically_contained(poly_a, poly_b, return_diff=False):
    """
    Check if polygon A is vertically contained within polygon B, ignoring X-axis spillover.

    Parameters:
    poly_a (Polygon): Polygon A.
    poly_b (Polygon): Polygon B.

    Returns:
    bool: True if A is vertically contained within B.
    """

    if isinstance(poly_a, Polygon):
        y_coords_a = [point[1] for point in poly_a.exterior.coords]
    elif isinstance(poly_a, MultiPolygon):
        # Choose the largest polygon inside the MultiPolygon
        largest = max(poly_a.geoms, key=lambda p: p.area)
        y_coords_a = [point[1] for point in largest.exterior.coords]
    else:
        raise TypeError(f"Unsupported geometry type: {type(poly_a)}")

    # y_coords_a = [point[1] for point in poly_a.exterior.coords]
    y_coords_b = [point[1] for point in poly_b.exterior.coords]

    # Check vertical containment along the Y-axis
    min_a, max_a = min(y_coords_a), max(y_coords_a)
    min_b, max_b = min(y_coords_b), max(y_coords_b)

    is_contained = min_b <= min_a and max_a <= max_b
    if return_diff:
        vertical_diff = (min_b + max_b) / 2 - (min_a + max_a) / 2
        return is_contained, vertical_diff
    else:
        return is_contained


def project_vector(vector, origin, normal):
    transform = trimesh.geometry.plane_transform(origin, normal)
    transformed = trimesh.transformations.transform_points(
        [np.array([0, 0, 0]), vector], transform
    )[:, :2]
    transformed_vector = transformed[1] - transformed[0]
    return transformed_vector


@gin.configurable
def stable_against(
    state: state_def.State,
    obj_name: str,
    relation_state: state_def.RelationState,
    visualize=False,
    allow_overhangs=False,
    use_initial=False,
    fix_pos=False,
):
    # 这个函数stable_against主要用于检查两个对象在三维空间中的稳定性。
    # 首先，它获取相关的对象和它们的平面表示。
    # 然后，函数检查两个对象的法向量是否平行，以确保它们是稳定的。
    # 接下来，它会将一个对象的网格投影到另一个对象的平面上，并根据允许的条件（例如是否允许悬垂）来判断它们是否重叠或相互包含。
    # 最后，函数还会验证每个顶点到目标平面的距离是否在允许的范围内。如果所有条件都满足，返回True，否则返回False。
    """
    check paralell, close to, and not overhanging.
    """
    # LAST

    """
    relation_state = RelationState(relation=StableAgainst({Subpart.Bottom, -Subpart.Top, -Subpart.Back, -Subpart.Front}, {Subpart.Visible, Subpart.SupportSurface, -Subpart.Ceiling, -Subpart.Wall}), target_name='bathroom_0-0', child_plane_idx=0, parent_plane_idx=0)
    """
    relation = relation_state.relation
    """
    relation = StableAgainst({Subpart.Bottom, -Subpart.Top, -Subpart.Back, -Subpart.Front}, {Subpart.Visible, Subpart.SupportSurface, -Subpart.Ceiling, -Subpart.Wall})

    """
    if obj_name == "5048934_BookStackFactory":
        a = 1
    assert isinstance(relation, cl.StableAgainst)

    logger.debug(f"stable against {obj_name=} {relation_state=}")
    a_blender_obj = state.objs[obj_name].obj

    if (
        Subpart.SupportSurface in relation_state.relation.parent_tags
        and relation_state.target_name != "newroom_0-0"
        and hasattr(state.objs[relation_state.target_name], "populate_obj")
    ):  # TODO YYD
        b_blender_obj = bpy.data.objects.get(
            state.objs[relation_state.target_name].populate_obj
        )
    else:
        b_blender_obj = state.objs[relation_state.target_name].obj

    # b_blender_obj = state.objs[relation_state.target_name].obj
    sa = state.objs[obj_name]
    sb = state.objs[relation_state.target_name]

    pa, pb = state.planes.get_rel_state_planes(state, obj_name, relation_state)

    poly_a = state.planes.planerep_to_poly(pa)
    poly_b = state.planes.planerep_to_poly(pb)

    normal_a = iu.global_polygon_normal(a_blender_obj, poly_a)
    normal_b = iu.global_polygon_normal(b_blender_obj, poly_b)
    dot = np.array(normal_a).dot(normal_b)
    if not (np.isclose(np.abs(dot), 1, atol=1e-2) or np.isclose(dot, -1, atol=1e-2)):
        logger.debug(f"stable against failed, not parallel {dot=}")
        return False

    origin_b = iu.global_vertex_coordinates(
        b_blender_obj, b_blender_obj.data.vertices[poly_b.vertices[0]]
    )

    scene = state.trimesh_scene

    if (
        Subpart.SupportSurface in relation_state.relation.parent_tags
        and relation_state.target_name != "newroom_0-0"
        and hasattr(state.objs[relation_state.target_name], "populate_obj")
    ):  # TODO YYD
        sb_obj = bpy.data.objects.get(
            state.objs[relation_state.target_name].populate_obj
        )
    else:
        sb_obj = state.objs[relation_state.target_name].obj

    a_trimesh = iu.meshes_from_names(scene, sa.obj.name)[0]
    b_trimesh = iu.meshes_from_names(scene, sb_obj.name)[0]

    mask = tagging.tagged_face_mask(sb_obj, relation.parent_tags)
    mask = state.planes.tagged_plane_mask(sb_obj, mask, pb)
    assert mask.any()
    b_trimesh_mask = b_trimesh.submesh([np.where(mask)[0]], append=True)

    # Project mesh A onto the plane of mesh B
    projected_a = trimesh.path.polygons.projected(a_trimesh, normal_b, origin_b)
    projected_b = trimesh.path.polygons.projected(b_trimesh_mask, normal_b, origin_b)
    logger.debug(
        f"stable_against projecting along {normal_b} for parent_tags {relation.parent_tags}"
    )

    if projected_a is None or projected_b is None:
        raise ValueError(f"Invalid {projected_a=} {projected_b=}")

    if allow_overhangs:
        res = projected_a.overlaps(projected_b)
    elif relation.check_z:
        res = projected_a.within(projected_b.buffer(1e-2))
        if (use_initial and not res) or (
            not use_initial and sa.dof_matrix_translation is not None and not res
        ):
            # if use_initial and not res and not fix_pos:
            move_vectors = []
            vertices = list(np.array(projected_a.exterior.coords))
            for point in vertices:
                p = Point(point)

                if projected_b.geom_type == "MultiPolygon":
                    flag_contains = False
                    for poly in projected_b.geoms:
                        if poly.contains(p):
                            flag_contains = True
                            break
                    if flag_contains:
                        continue
                elif projected_b.contains(p):
                    continue

                if projected_b.geom_type == "MultiPolygon":
                    closest_points = [
                        poly.exterior.interpolate(poly.exterior.project(p))
                        for poly in projected_b.geoms
                    ]
                    closest_point = min(
                        closest_points, key=lambda pt: pt.distance(p)
                    )  # Find the nearest one
                else:
                    closest_point = projected_b.exterior.interpolate(
                        projected_b.exterior.project(p)
                    )
                move_vector = [closest_point.x - p.x, closest_point.y - p.y]
                move_vectors.append(
                    np.array(move_vector)[None, :]
                    / np.linalg.norm(np.array(move_vector))
                )
            if move_vectors != []:
                gradient = np.mean(np.concatenate(move_vectors, axis=0), axis=0)
                if obj_name == "60910_nightstand":
                    a = 1
                gradient = anti_project_to_3d(gradient, normal_b)
                gradient = sa.dof_matrix_translation @ gradient
                gradient_norm = np.linalg.norm(gradient)
                if gradient_norm == 0:
                    translation = np.zeros(3)
                else:
                    TRANS_MULT = 0.05  # min(0.05,gradient_norm)
                    translation = TRANS_MULT * gradient / gradient_norm
                # gradient = [gradient[0],gradient[1],0]
                # TRANS_MULT = 0.05
                # translation = TRANS_MULT * sa.dof_matrix_translation @ gradient
                iu.translate(state.trimesh_scene, sa.obj.name, translation)
                print(obj_name, bpy.data.objects[sa.obj.name].location)

    else:
        if obj_name == "6569851_FloorLampFactory":
            a = 1
        z_proj = project_vector(np.array([0, 0, 1]), origin_b, normal_b)
        projected_a_rotated, projected_b_rotated = project_and_align_z_with_x(
            [projected_a, projected_b], z_proj
        )
        res, vertical_diff = is_vertically_contained(
            projected_a_rotated, projected_b_rotated, return_diff=True
        )
        if use_initial and not res and not fix_pos:
            tangent_1 = [0, 0, 1]
            tangent_2 = np.cross(normal_b, tangent_1)
            tangent_2 = tangent_2 / np.linalg.norm(tangent_2)
            gradient = vertical_diff * tangent_2

            gradient = sa.dof_matrix_translation @ gradient
            # gradient[2] = 0
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm == 0:
                translation = np.zeros(3)
            else:
                TRANS_MULT = 0.05  # #min(0.05,gradient_norm)
                translation = TRANS_MULT * gradient / gradient_norm

            # TRANS_MULT = 0.1
            # translation = TRANS_MULT * sa.dof_matrix_translation @ gradient

            iu.translate(state.trimesh_scene, sa.obj.name, translation)

    if visualize:
        fig, ax = plt.subplots()
        iu.plot_geometry(ax, projected_a, "blue")
        iu.plot_geometry(ax, projected_b, "green")
        plt.title(f"{obj_name} stable against {relation_state.target_name}? {res=}")
        plt.show()

    logger.debug(f"stable_against {res=}")
    if not res:
        return False

    for vertex in poly_a.vertices:
        vertex_global = iu.global_vertex_coordinates(
            a_blender_obj, a_blender_obj.data.vertices[vertex]
        )
        distance = iu.distance_to_plane(vertex_global, origin_b, normal_b)
        if not np.isclose(distance, relation_state.relation.margin, atol=1e-2):
            logger.debug(f"stable against failed, not close to {distance=}")
            return False

    return True


def anti_project_to_3d(point_2D, normal_b, origin_b=[0, 0, 0]):
    normal_b = np.array(normal_b)
    # Find two vectors that lie on the plane, perpendicular to the normal vector
    # Using cross products to find perpendicular vectors
    # tangent_1 = np.cross(normal_b, [1, 0, 0]) if normal_b[0] != 0 else np.cross(normal_b, [0, -1, 0])

    if normal_b[1] not in [1, -1]:
        tangent_1 = np.cross(normal_b, [0, -1, 0])
    elif normal_b[1] not in [1, -1]:
        tangent_1 = np.cross(normal_b, [0, 1, 0])
    # # else:
    # tangent_1 = np.cross(normal_b, [1, 0, 0]) if normal_b != [1,0,0] else np.cross(normal_b, [0, -1, 0])
    # tangent_1 = tangent_1 / np.linalg.norm(tangent_1)

    if normal_b[1] not in [1, -1]:
        # If normal is vertical (aligned with Y-axis), use X and Z
        tangent_1 = np.array([1, 0, 0], dtype=np.float64)
    else:
        # Otherwise, cross with world Y axis
        tangent_1 = np.cross(normal_b, [0, 1, 0])
        tangent_1 /= np.linalg.norm(tangent_1)

    # Find another tangent vector
    tangent_2 = np.cross(normal_b, tangent_1)
    tangent_2 = tangent_2 / np.linalg.norm(tangent_2)

    # Now you can reconstruct the 3D point corresponding to the 2D point
    point_3D = origin_b + point_2D[0] * tangent_1 + point_2D[1] * tangent_2

    return point_3D


def snap_against(scene, a, b, a_plane, b_plane, margin=0, rev_normal=False):
    """
    snap a against b with some margin.
    将对象 `a` 对齐到对象 `b` 的指定平面上，同时允许指定一个间距（margin）。
    参数：
    - scene: Blender 场景对象，用于操作。
    - a: 对象 `a` 的名称。
    - b: 对象 `b` 的名称。
    - a_plane: 对象 `a` 的平面定义，包含平面索引信息。
    - b_plane: 对象 `b` 的平面定义，包含平面索引信息。
    - margin: 对齐时的间距（单位为 Blender 的单位）。
    """
    logging.debug("snap_against", a, b, a_plane, b_plane, margin, rev_normal)

    a_obj = bpy.data.objects[a]
    b_obj = bpy.data.objects[b]
    # 从对象 `a` 和 `b` 的数据中提取平面（多边形）信息。
    a_poly_index = a_plane[1]
    a_poly = a_obj.data.polygons[a_poly_index]
    b_poly_index = b_plane[1]
    b_poly = b_obj.data.polygons[b_poly_index]
    # 获取平面 `a` 和 `b` 上顶点的全局坐标。
    plane_point_a = butil.global_vertex_coordinates(
        a_obj, a_obj.data.vertices[a_poly.vertices[0]]
    )
    # 获取平面 `a` 和 `b` 的法向量。
    plane_normal_a = butil.global_polygon_normal(a_obj, a_poly)
    plane_point_b = butil.global_vertex_coordinates(
        b_obj, b_obj.data.vertices[b_poly.vertices[0]]
    )
    plane_normal_b = butil.global_polygon_normal(b_obj, b_poly, rev_normal)
    plane_normal_b = -plane_normal_b
    # 确保平面法向量的长度为 1，否则报错。
    norm_mag_a = np.linalg.norm(plane_normal_a)
    norm_mag_b = np.linalg.norm(plane_normal_b)
    assert np.isclose(norm_mag_a, 1), norm_mag_a
    assert np.isclose(norm_mag_b, 1), norm_mag_b
    # 计算 `a` 和 `b` 的法向量之间的旋转轴。
    rotation_axis = np.cross(plane_normal_a, plane_normal_b)
    if not np.isclose(np.linalg.norm(rotation_axis), 0, atol=1e-05):
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    else:
        rotation_axis = np.array([0, 0, 1])
    # 计算两个法向量之间的夹角。
    dot = plane_normal_a.dot(plane_normal_b)
    rotation_angle = np.arccos(np.clip(dot, -1, 1))
    if np.isnan(rotation_angle):
        raise ValueError(f"Invalid {rotation_angle=}")
    # 旋转对象 `a`，使其法向量与 `b` 的法向量对齐。
    iu.rotate(scene, a, rotation_axis, rotation_angle)

    # 更新对象 `a` 和其法向量信息（旋转后需重新计算）。
    a_obj = bpy.data.objects[a]
    a_poly = a_obj.data.polygons[a_poly_index]
    # Recalculate vertex_a and normal_a after rotation
    plane_point_a = butil.global_vertex_coordinates(
        a_obj, a_obj.data.vertices[a_poly.vertices[0]]
    )
    plane_normal_a = butil.global_polygon_normal(a_obj, a_poly)
    # 计算平面 `a` 和 `b` 之间的距离（沿 `b` 平面法向量方向）。
    distance = (plane_point_a - plane_point_b).dot(plane_normal_b)
    # 计算平移向量，将对象 `a` 移动到与 `b` 对齐的目标位置，并考虑指定的 `margin`。
    # Move object a by the average distance minus the margin in the direction of the plane normal of b
    translation = -(distance + margin) * plane_normal_b.normalized()
    iu.translate(scene, a, translation)


def random_sample_point(
    state: state_def.State,
    obj: bpy.types.Object,
    face_mask: np.ndarray,
    plane: tuple[str, int],
) -> Vector:
    """
    Given a plane, return a random point on the plane.
    """

    if obj.type != "MESH":
        raise ValueError(f"Unexpected {obj.type=}")

    plane_mask = state.planes.tagged_plane_mask(obj, face_mask, plane)
    if not np.any(plane_mask):
        logging.warning(
            f"No faces in object {obj.name} are coplanar with plane {plane}."
        )

    # Create a bmesh from the object mesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    faces = [bm.faces[i] for i in np.where(plane_mask)[0]]

    # Calculate the area for each face and create a cumulative distribution
    areas = np.array([f.calc_area() for f in faces])
    cumulative_areas = np.cumsum(areas)
    total_area = cumulative_areas[-1]

    # Generate a random number and find the corresponding face
    random_area_point = np.random.rand() * total_area
    face_index = np.searchsorted(cumulative_areas, random_area_point)
    selected_face = faces[face_index]

    verts = [v.co for v in selected_face.verts]

    # Use barycentric coordinates to sample a random point in the triangle
    # Random weights for each vertex
    weights = np.random.rand(3)
    weights /= np.sum(weights)
    random_point_local = (
        weights[0] * verts[0] + weights[1] * verts[1] + weights[2] * verts[2]
    )
    random_point_global = obj.matrix_world @ Vector(random_point_local)

    bm.free()

    return random_point_global


def move_obj_random_pt(
    state: state_def.State, a, b, face_mask: np.ndarray, plane: tuple[str, int]
):
    """
    move a to a random point on b
    """
    scene = state.trimesh_scene
    b_obj = iu.blender_objs_from_names(b)[0]

    random_point_global = random_sample_point(state, b_obj, face_mask, plane)

    iu.set_location(scene, a, random_point_global)


# def place_randomly(scene, a, b, visualize = False):
#     """
#     place a randomly on b.
#     """
#     a_blender_mesh  = blender_objs_from_names(a)[0]
#     a_trimesh = meshes_from_names(scene, a)[0]
#     b_blender_mesh = blender_objs_from_names(b)[0]
#     b_trimesh = meshes_from_names(scene, b)[0]
#     b_proj = project_to_xy_poly(b_trimesh)

#     xy_loc = sample_random_point(b_proj)
#     if visualize:
#         fig, ax = plt.subplots()
#         if isinstance(b_proj, Polygon):
#             x, y = b_proj.exterior.xy
#             ax.fill(x, y, alpha=0.5, fc='red', ec='black', label='Polygon b')
#         elif isinstance(b_proj, MultiPolygon):
#             for sub_poly in b_proj.geoms:
#                 x, y = sub_poly.exterior.xy
#                 ax.fill(x, y, alpha=0.5, fc='red', ec='black', label='Polygon b')
#         ax.plot(xy_loc.x, xy_loc.y, 'o', color='black', label='Random point')
#         plt.show()

#     set_location(scene, a, Vector((xy_loc.x, xy_loc.y, 0)))


def supported_by(scene, a, b, visualize=False):
    # check for collision first

    if isinstance(a, str):
        a = [a]

    a_meshes = iu.blender_objs_from_names(a)
    a_trimeshes = iu.meshes_from_names(scene, a)
    b_mesh = iu.blender_objs_from_names(b)[0]
    b_trimesh = iu.meshes_from_names(scene, b)[0]

    if visualize:
        fig, ax = plt.subplots()
        ax.set_aspect("equal", "box")
        b_poly = iu.project_to_xy_poly(b_trimesh)
        if isinstance(b_poly, Polygon):
            x, y = b_poly.exterior.xy
            ax.fill(x, y, alpha=0.5, fc="red", ec="black", label="Polygon b")
        elif isinstance(b_poly, MultiPolygon):
            for sub_poly in b_poly.geoms:
                x, y = sub_poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc="red", ec="black", label="Polygon b")

    for a_mesh, a_trimesh in zip(a_meshes, a_trimeshes):
        cloned_a = butil.deep_clone_obj(
            a_mesh, keep_modifiers=True, keep_materials=False
        )
        butil.modify_mesh(
            cloned_a, "BOOLEAN", apply=True, operation="INTERSECT", object=b_mesh
        )
        iu.preprocess_obj(cloned_a)
        intersection = iu.to_trimesh(cloned_a)
        intersection_poly = iu.project_to_xy_poly(intersection)
        intersection_convex = intersection_poly.convex_hull
        com_projected = a_trimesh.centroid[:2]
        if visualize:
            if isinstance(intersection_poly, Polygon):
                x, y = intersection_poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc="blue", ec="black", label="Polygon a")
            elif isinstance(intersection_poly, MultiPolygon):
                for sub_poly in intersection_poly.geoms:
                    x, y = sub_poly.exterior.xy
                    ax.fill(x, y, alpha=0.5, fc="blue", ec="black", label="Polygon a")
            ax.plot(
                com_projected[0], com_projected[1], "o", color="black", label="COM of a"
            )

        if not intersection_convex.contains(Point(com_projected)):
            if visualize:
                plt.show()
            return False
    if visualize:
        plt.show()
    return True
