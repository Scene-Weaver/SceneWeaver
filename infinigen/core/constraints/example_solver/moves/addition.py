# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import logging
import typing
from dataclasses import dataclass

import bpy
import numpy as np

from infinigen.assets.utils import bbox_from_mesh
from infinigen.core import tagging
from infinigen.core import tags as t
from infinigen.core.constraints import usage_lookup
from infinigen.core.constraints.constraint_language import util as iu
from infinigen.core.constraints.constraint_language.util import delete_obj
from infinigen.core.constraints.example_solver.geometry import (
    dof,
    parse_scene,
    validity,
)
from infinigen.core.constraints.example_solver.state_def import ObjectState, State
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen_examples.util.visible import invisible_others, visible_others

from . import moves
from .reassignment import pose_backup, restore_pose_backup
import os

# from line_profiler import LineProfiler


logger = logging.getLogger(__name__)

GLOBAL_GENERATOR_SINGLETON_CACHE = {}


def sample_rand_placeholder(gen_class: type[AssetFactory], dimension=None):
    singleton_gen = usage_lookup.has_usage(gen_class, t.Semantics.SingleGenerator)

    if singleton_gen and gen_class in GLOBAL_GENERATOR_SINGLETON_CACHE:
        gen = GLOBAL_GENERATOR_SINGLETON_CACHE[gen_class]
    else:
        fac_seed = np.random.randint(1e7)
        gen = gen_class(fac_seed)
        if singleton_gen:
            GLOBAL_GENERATOR_SINGLETON_CACHE[gen_class] = gen

    # if dimension is not None:
    #     gen.params['Dimensions'] = dimension

    inst_seed = np.random.randint(1e7)
    # MARK placeholder
    if usage_lookup.has_usage(gen_class, t.Semantics.RealPlaceholder):
        new_obj = gen.spawn_placeholder(
            inst_seed, loc=(0, 0, 0), rot=(0, 0, 0)
        )  # generate bbox with dimension
    elif usage_lookup.has_usage(gen_class, t.Semantics.AssetAsPlaceholder):
        new_obj = gen.spawn_asset(
            inst_seed, loc=(0, 0, 0), rot=(0, 0, 0)
        )  # generate asset
    elif usage_lookup.has_usage(gen_class, t.Semantics.PlaceholderBBox):
        new_obj = bbox_from_mesh.bbox_mesh_from_hipoly(gen, inst_seed, use_pholder=True)
    else:
        new_obj = bbox_from_mesh.bbox_mesh_from_hipoly(
            gen, inst_seed
        )  # generate bbox with asset

    if new_obj.type != "MESH":
        raise ValueError(f"Addition created {new_obj.name=} with type {new_obj.type}")
    if len(new_obj.data.polygons) == 0:
        raise ValueError(f"Addition created {new_obj.name=} with 0 faces")

    butil.put_in_collection(
        list(butil.iter_object_tree(new_obj)), butil.get_collection("placeholders")
    )
    parse_scene.preprocess_obj(new_obj)
    tagging.tag_canonical_surfaces(new_obj)

    return new_obj, gen


@dataclass
class Addition(moves.Move):
    """Move which generates an object and adds it to the scene with certain relations"""

    gen_class: typing.Any
    relation_assignments: list
    temp_force_tags: set

    _new_obj: bpy.types.Object = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.gen_class.__name__}, {len(self.relation_assignments)} relations)"
        # Addition(BedFactory, 2 relations)

    def apply(self, state: State, expand_collision=False):  # mark
        (target_name,) = self.names

        #     import pdb
        #     pdb.set_trace()
        assert target_name not in state.objs

        # self._new_obj, gen = sample_rand_placeholder(self.gen_class)
        self._new_obj, gen = sample_rand_placeholder(
            self.gen_class, dimension=(1, 1, 1)
        )

        # center = np.array([v.co for v in self._new_obj.data.vertices]).mean(axis=0)
        # size = self._new_obj.dimensions

        parse_scene.add_to_scene(state.trimesh_scene, self._new_obj, preprocess=True)

        tags = self.temp_force_tags.union(usage_lookup.usages_of_factory(gen.__class__))

        assert isinstance(self._new_obj, bpy.types.Object)
        objstate = ObjectState(
            obj=self._new_obj,
            generator=gen,
            tags=tags,
            relations=self.relation_assignments,
        )

        state.objs[target_name] = objstate

        # if target_name == "113239_OfficeChairFactory":
        #     a = 1
        # if target_name != "113239_OfficeChairFactory" and  "OfficeChairFactory" in target_name:
        #     a = 1
        #     bpy.ops.wm.save_as_mainfile(filepath="/home/yandan/Desktop/a.blend")

        success = dof.try_apply_relation_constraints(
            state, target_name, expand_collision=expand_collision
        )  # check
        # if success:
        #     if (
        #         "LargeShelfFactory(1502912).bbox_placeholder(2697479)"
        #         in objstate.obj.name
        #     ):
        #         import pdb

        #         pdb.set_trace()
        #     a = 1
        logger.debug(f"{self} {success=}")
        return success

    def apply_init(
        self,
        state: State,
        target_name,
        size,
        position,
        rotation,
        gen_class,
        asset_file=None,
        expand_collision=False,
    ):  # mark
        assert target_name not in state.objs
        import copy

        self._new_obj, gen = sample_rand_placeholder(gen_class)
        
        if size is not None:
            self._new_obj = resize_obj(self._new_obj, size)

        parse_scene.add_to_scene(state.trimesh_scene, self._new_obj, preprocess=True)

        tags = self.temp_force_tags.union(usage_lookup.usages_of_factory(gen.__class__))

        assert isinstance(self._new_obj, bpy.types.Object)
        objstate = ObjectState(
            obj=self._new_obj,
            generator=gen,
            tags=tags,
            size=size,
            relations=self.relation_assignments,
        )

        state.objs[target_name] = objstate

        # name = "SofaFactory(1351066).bbox_placeholder(2179127)"
        name = self._new_obj.name

        if gen_class.__name__ == "MetaCategoryFactory":
            rotation = gen.rotation_orig
            position = gen.location_orig
            room_width = os.getenv("room_width")
            room_height = os.getenv("room_height")
            position[0] += int(room_width)/2
            position[1] += int(room_height)/2

        # if gen_class.__name__ == "ThreedFrontCategoryFactory":
        #     rotation = gen.rotation_orig
        #     position = gen.location_orig
            

        # invisible_others()
        # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        # visible_others()
        
                

        iu.translate(state.trimesh_scene, name, position)
        iu.rotate(state.trimesh_scene, name, np.array([0, 0, 1]), rotation)


        success = dof.try_apply_relation_constraints(
            state,
            target_name,
            expand_collision=expand_collision,
            n_try_resolve=1,
            use_initial=True,
            closest_surface=True,
        )  # check
        logger.debug(f"{self} {success=}")
        return success

        # save_path = "debug.blend"
        # bpy.ops.wm.save_as_mainfile(filepath=save_path)

    def revert(self, state: State):
        to_delete = list(butil.iter_object_tree(self._new_obj))
        delete_obj(state.trimesh_scene, [a.name for a in to_delete])

        (new_name,) = self.names
        del state.objs[new_name]


def resize_obj(obj, size, apply_transform=True):
    
    x_dim, y_dim, z_dim = size
    if x_dim==-1:
        return obj
    x_scale = x_dim / obj.dimensions[0]
    y_scale = y_dim / obj.dimensions[1]
    z_scale = z_dim / obj.dimensions[2]

    obj.scale = (x_scale, y_scale, z_scale)

    if apply_transform:
        butil.apply_transform(obj, True)

    return obj


@dataclass
class Resample(moves.Move):
    """Move which replaces an existing object with a new one from the same generator"""

    align_corner: int = None

    _backup_gen = None
    _backup_obj = None
    _backup_poseinfo = None

    def apply(self, state: State, expand_collision=False):
        assert len(self.names) == 1
        target_name = self.names[0]

        os = state.objs[target_name]
        self._backup_gen = os.generator
        self._backup_obj = os.obj
        self._backup_poseinfo = pose_backup(os)

        scene = state.trimesh_scene
        scene.graph.transforms.remove_node(os.obj.name)
        scene.delete_geometry(os.obj.name + "_mesh")

        os.obj, os.generator = sample_rand_placeholder(os.generator.__class__)

        if self.align_corner is not None:
            self._backup_obj.bound_box[self.align_corner]
            os.obj.bound_box[self.align_corner]
            raise NotImplementedError(f"{self.align_corner=}")

        parse_scene.add_to_scene(state.trimesh_scene, os.obj, preprocess=True)
        dof.apply_relations_surfacesample(state, target_name)

        return validity.check_post_move_validity(
            state, target_name, expand_collision=expand_collision
        )

    def revert(self, state: State):
        (target_name,) = self.names

        os = state.objs[target_name]
        delete_obj(state.trimesh_scene, os.obj.name)

        os.obj = self._backup_obj
        os.generator = self._backup_gen
        parse_scene.add_to_scene(state.trimesh_scene, os.obj, preprocess=False)
        restore_pose_backup(state, target_name, self._backup_poseinfo)

    def accept(self, state: State):
        butil.delete(list(butil.iter_object_tree(self._backup_obj)))
