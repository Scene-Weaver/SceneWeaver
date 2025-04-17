# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Alexander Raistrick: AssetFactory, make_asset_collection
# - Lahav Lipson: quickly_resample


import logging
import typing

import bpy
import numpy as np
from tqdm import trange

from infinigen.assets.utils.object import center
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, int_hash
import mathutils
from . import detail

logger = logging.getLogger(__name__)


class AssetFactory:
    def __init__(self, factory_seed=None, coarse=False):
        self.factory_seed = factory_seed
        if self.factory_seed is None:
            self.factory_seed = np.random.randint(1e9)

        self.coarse = coarse

        logger.debug(f"{self}.__init__()")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.factory_seed})"

    @staticmethod
    def quickly_resample(obj):
        assert obj.type == "EMPTY", obj.type
        obj.rotation_euler[2] = np.random.uniform(-np.pi, np.pi)

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        # Optionally, override this function to decide what will be used as a placeholder for your asset
        return butil.spawn_cube(size=2)

    def finalize_placeholders(self, placeholders: typing.List[bpy.types.Object]):
        # Optionally, override this function to perform any operations on all the placeholders at once
        # eg joint space colonization, placing vines between trees
        pass

    def asset_parameters(self, distance: float, vis_distance: float) -> dict:
        # Optionally, override to determine the **params input of create_asset w.r.t. camera distance
        return {
            "face_size": detail.target_face_size(distance),
            "distance": distance,
            "vis_distance": vis_distance,
        }

    def create_asset(self, **params) -> bpy.types.Object:
        # Override this function to produce a high detail asset
        raise NotImplementedError

    def finalize_assets(self, assets):
        # Optionally, override this function to perform any operations on all the assets at once
        # eg any cleanup / grouping
        pass

    def spawn_placeholder(self, i, loc, rot):
        # 生成一个占位符对象（placeholder），并根据条件进行位置、旋转等属性的调整
        # Not intended to be overridden - override create_placeholder instead
        # 函数内还会对某些特殊约束（例如沿路径跟随的约束）做处理，确保对象的正确位置。

        logger.debug(f"{self}.spawn_placeholder({i}...)")
        # MARK
        with FixedSeed(int_hash((self.factory_seed, i))):
            obj = self.create_placeholder(i=i, loc=loc, rot=rot)  # MARK size

        has_sensitive_constraint = any(
            c.type in ["FOLLOW_PATH"] for c in obj.constraints
        )
        if not has_sensitive_constraint:
            obj.location = loc
            obj.rotation_euler = rot
        else:
            logger.debug(
                f"Not assigning placeholder {obj.name=} location due to presence of"
                "location-sensitive constraint, typically a follow curve"
            )
        obj.name = f"{repr(self)}.spawn_placeholder({i})"  #'BedFactory(5615189).spawn_placeholder(2274623)'

        if obj.parent is not None:
            logger.warning(
                f"{obj.name=} has no-none parent {obj.parent.name=}, this may cause it not to get populated"
            )

        return obj

    def spawn_asset(
        self,
        i,
        placeholder=None,
        distance=None,
        vis_distance=0,
        loc=(0, 0, 0),
        rot=(0, 0, 0),
        **kwargs,
    ):
        if not isinstance(i, int):
            raise TypeError(f"{i=} {type(i)=}, expected int")
        # Not intended to be overridden - override create_asset instead

        logger.debug(f"{self}.spawn_asset({i}...)")

        if distance is None:
            distance = detail.scatter_res_distance()

        if self.coarse:
            raise ValueError(
                "Attempted to spawn_asset() on an AssetFactory(coarse=True)"
            )

        # TODO
        user_provided_placeholder = placeholder is not None

        if user_provided_placeholder:
            assert loc == (0, 0, 0) and rot == (0, 0, 0)
        else:
            placeholder = self.spawn_placeholder(i=i, loc=loc, rot=rot)
            self.finalize_placeholders([placeholder])
            # placeholder
            # center = np.array([v.co for v in placeholder.data.vertices]).mean(axis=0)
            # size = placeholder.dimensions

        # 定义需要进行垃圾回收的目标对象列表
        gc_targets = [
            bpy.data.meshes,
            bpy.data.textures,
            bpy.data.node_groups,
            bpy.data.materials,
        ]

        with (
            FixedSeed(
                int_hash((self.factory_seed, i))
            ),  # 固定随机种子，确保每次生成结果相同
            butil.GarbageCollect(gc_targets, verbose=False),  # 垃圾回收
        ):
            # 获取资产的参数，这些参数用于资产生成
            params = self.asset_parameters(distance, vis_distance)
            params.update(kwargs)
            obj = self.create_asset(i=i, placeholder=placeholder, **params)

        obj.name = f"{repr(self)}.spawn_asset({i})"

        if user_provided_placeholder:
            if obj is not placeholder:
                if obj.parent is None:
                    butil.parent_to(obj, placeholder, no_inverse=True)
            else:
                obj.hide_render = False
        else:
            obj.parent = None
            obj.location = placeholder.location
            obj.rotation_euler = placeholder.rotation_euler
            butil.delete(placeholder)

        return obj

    __call__ = spawn_asset  # for convinience

    def post_init(self):
        pass

    def set_origin(self,imported_obj):
        imported_obj.location = [0,0,0]
        bbox_corners = [mathutils.Vector(corner) for corner in imported_obj.bound_box]

        min_z = min(corner.z for corner in bbox_corners)
        imported_obj.location.z -= min_z

        mean_x = np.mean([corner.x for corner in bbox_corners])
        imported_obj.location.x -= mean_x
        mean_y = np.mean([corner.y for corner in bbox_corners])
        imported_obj.location.y -= mean_y

        pos_bias = [mean_x,mean_y,min_z]

        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='BOUNDS')
        return imported_obj,pos_bias



def make_asset_collection(
    spawn_fns,
    n,
    name=None,
    weights=None,
    as_list=False,
    verbose=True,
    centered=False,
    **kwargs,
):
    if not isinstance(spawn_fns, list):
        spawn_fns = [spawn_fns]
    if weights is None:
        weights = np.ones(len(spawn_fns))
    weights /= sum(weights)

    if name is None:
        name = ",".join([repr(f) for f in spawn_fns])

    if verbose:
        logger.info(f"Generating collection of {n} assets from {name}")

    objs = [[] for _ in range(len(spawn_fns))]
    r = trange(n) if verbose else range(n)
    for i in r:
        fn_idx = np.random.choice(np.arange(len(spawn_fns)), p=weights)
        obj = spawn_fns[fn_idx](i=i, **kwargs)
        if centered:
            obj.location = -center(obj)
            butil.apply_transform(obj, True)
        objs[fn_idx].append(obj)

    for os, f in zip(objs, spawn_fns):
        if hasattr(f, "finalize_assets"):
            f.finalize_assets(os)

    objs = sum(objs, start=[])

    if as_list:
        return objs
    else:
        col = butil.group_in_collection(objs, name=f"assets:{name}", reuse=False)
        col.hide_viewport = True
        col.hide_render = True
        return col
