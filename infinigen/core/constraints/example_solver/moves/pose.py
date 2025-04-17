# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick, Karhan Kayan

import logging
from dataclasses import dataclass

import bpy
import numpy as np

from infinigen.core.constraints.constraint_language import util as iu
from infinigen.core.constraints.example_solver.geometry import dof, validity
from infinigen.core.constraints.example_solver.geometry.dof import (
    apply_relations_surfacesample,
    combine_rotation_constraints,
    combined_stability_matrix,
)
from infinigen.core.constraints.example_solver.state_def import State
from infinigen_examples.util.visible import invisible_others, visible_others

from . import moves
from .reassignment import pose_backup, restore_pose_backup
from infinigen.core.tags import Subpart

logger = logging.getLogger(__name__)


@dataclass
class TranslateMove(moves.Move):
    # translate obj by vector

    translation: np.array

    _backup_pose: dict = None

    def __repr__(self):
        norm = np.linalg.norm(self.translation)
        return f"{self.__class__.__name__}({self.names}, {norm:.2e})"

    def apply(self, state: State, expand_collision=False):
        (target_name,) = self.names

        os = state.objs[target_name]
        self._backup_pose = pose_backup(os, dof=False)

        iu.translate(state.trimesh_scene, os.obj.name, self.translation)

        if not validity.check_post_move_validity(
            state, target_name, expand_collision=expand_collision
        ):
            return False
        # if (
        #     "LargeShelfFactory(1502912).bbox_placeholder(2697479)"
        #     in state.objs[target_name].obj.name
        # ):
        #     import pdb

        #     pdb.set_trace()
        return True

    def revert(self, state: State):
        (target_name,) = self.names
        restore_pose_backup(state, target_name, self._backup_pose)

    def apply_gradient(self, state: State, temperature=None, expand_collision=False):
        (target_name,) = self.names

        # state.trimesh_scene.show()
        os = state.objs[target_name]

        obj_state = state.objs[target_name]
        # print(target_name,s "1 ",obj_state.obj.location)

        parent_planes = apply_relations_surfacesample(
            state, target_name, use_initial=True,closest_surface=True
        )
        obj_state.dof_matrix_translation = combined_stability_matrix(
            parent_planes
        )  # 平移自由度的合成约束矩阵。
        obj_state.dof_rotation_axis = combine_rotation_constraints(
            parent_planes
        )  # 旋转自由度的约束轴或限制信息
        # print(target_name, "2 ",obj_state.obj.location)
        # result = validity.check_post_move_validity(
        result = validity.move_for_relation_and_collision(
            state,
            target_name,
            expand_collision=expand_collision,
            return_touch=True,
            use_initial=True,
        )

        # print(target_name, "3 ",obj_state.obj.location)
        success, touch = result

        self._backup_pose = pose_backup(os, dof=False)

        # invisible_others()
        # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        # visible_others()
        # import pdb
        # pdb.set_trace()
        if touch is None:
            # relation invalid, have moved to make it valid
            return False
        if not touch.hit:
            # do not touch
            return False
        # if isinstance(touch.names[1], str):
        #     # no collision
        #     return False
        if len(touch.names)==0:
            # no collision
            return False
        # if "ChairFactory" in target_name:
        #     a = 1
        #     TRANS_MULT = 8
        #     TRANS_MIN = 0.01
        #     var = max(TRANS_MIN, TRANS_MULT * temperature)
        #     random_vector = np.random.normal(0, var, size=3)

        #     self.translation = obj_state.dof_matrix_translation @ random_vector
        # else:
        self.translation = self.calc_gradient(
            state.trimesh_scene, state, target_name, touch
        )
        iu.translate(state.trimesh_scene, os.obj.name, self.translation)
        # print(target_name, "4 ",obj_state.obj.location)
        # print(target_name,self.translation)
        self._backup_pose = pose_backup(os, dof=False)
        # invisible_others()
        # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        # visible_others()
        # pdb.set_trace()
        return success
    

    def calc_gradient(self, scene, state, name, touch):

        obj_state = state.objs[name]
       
        # record top children
        childnames = set()
        for k,os in state.objs.items():
            for rel in os.relations:
                if rel.target_name==name and \
                    (Subpart.SupportSurface in rel.relation.parent_tags or Subpart.Top in rel.relation.parent_tags):
                    childnames.add(os.obj.name)

        a = obj_state.obj.name
        b_names = []
        gradient = np.zeros(3)
        valid_names = [x for x in touch.names if x!=a]
        
        # for _, b in touch.names:
        for i in range(len(valid_names)):
            b = valid_names[i]
            normal = touch.contacts[i].normal
            depth = touch.contacts[i].depth

            if b in childnames or b==state.objs[name].obj.name: # remove top children's collision
                continue
      
            if b not in b_names:
                b_names.append(b)
                gradient += normal*depth

        gradient[2] = 0
        gradient_norm = np.linalg.norm(gradient)
        if len(b_names) == 0 or gradient_norm==0:
            gradient = np.zeros(3)
        else:
            gradient = gradient / gradient_norm
        TRANS_MULT = 0.1
        translation = TRANS_MULT * obj_state.dof_matrix_translation @ gradient

        return translation

    # def calc_gradient(self, scene, state, name, touch):
    #     if name=="1753306_BookStackFactory":
    #         import pdb
    #     #     pdb.set_trace()
    #     obj_state = state.objs[name]
       
    #     # record top children
    #     childnames = set()
    #     for k,os in state.objs.items():
    #         for rel in os.relations:
    #             if rel.target_name==name and \
    #                 (Subpart.SupportSurface in rel.relation.parent_tags or Subpart.Top in rel.relation.parent_tags):
    #                 childnames.add(os.obj.name)

    #     a = obj_state.obj.name
    #     T, g = scene.graph[a]  # 获取 b 的变换和几何信息
    #     geom_a = scene.geometry[g]
    #     centroid_a = geom_a.centroid

    #     centroid_b_lst = []
    #     b_names = []
    #     # for _, b in touch.names:
    #     for b in touch.names:
    #         if b in childnames or b==state.objs[name].obj.name: # remove top children's collision
    #             continue
    #         T, g = scene.graph[b]  # 获取 b 的变换和几何信息
    #         geom_b = scene.geometry[g]
    #         centroid_b = geom_b.centroid
    #         if b not in b_names:
    #             b_names.append(b)
    #             centroid_b_lst.append(centroid_b)

    #     if centroid_b_lst==[]:
    #         return [0,0,0]
        
    #     centroid_b_mean = np.mean(centroid_b_lst, axis=0)
    #     if "FloorLampFactory" in name:
    #         a = 1

    #     gradient = centroid_a - centroid_b_mean
    #     gradient_norm = np.linalg.norm(gradient)
    #     gradient = gradient / gradient_norm
    #     TRANS_MULT = 0.05
    #     translation = TRANS_MULT * obj_state.dof_matrix_translation @ gradient

    #     return translation


@dataclass
class RotateMove(moves.Move):
    axis: np.array
    angle: float

    _backup_pose = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.names}, {self.angle:.2e})"

    def apply(self, state: State, expand_collision=False):
        (target_name,) = self.names

        os = state.objs[target_name]
        self._backup_pose = pose_backup(os, dof=False)

        iu.rotate(state.trimesh_scene, os.obj.name, self.axis, self.angle)

        if not validity.check_post_move_validity(
            state, target_name, expand_collision=expand_collision
        ):
            return False

        return True

    def revert(self, state: State):
        (target_name,) = self.names
        restore_pose_backup(state, target_name, self._backup_pose)


@dataclass
class ReinitPoseMove(moves.Move):
    _backup_pose: dict = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.names})"

    def apply(self, state: State, expand_collision=False):
        (target_name,) = self.names
        ostate = state.objs[target_name]
        self._backup_pose = pose_backup(ostate)
        return dof.try_apply_relation_constraints(
            state, target_name, expand_collision=expand_collision
        )

    def revert(self, state: State):
        (target_name,) = self.names
        restore_pose_backup(state, target_name, self._backup_pose)


"""
@dataclass
class ScaleMove(Move):
    name: str
    scale: np.array

    def apply(self, state: State):
        blender_obj = self.obj.bpy_obj
        trimesh_obj = state.get_trimesh_object(self.obj.name)
        blender_obj.scale *= Vector(self.scale)
        trimesh_obj.apply_transform(trimesh.transformations.compose_matrix(scale=list(self.scale)))
        self.obj.update()

    def revert(self, state: State):
        blender_obj = self.obj.bpy_obj
        trimesh_obj = state.get_trimesh_object(self.obj.name)
        blender_obj.scale /= Vector(self.scale)
        trimesh_obj.apply_transform(trimesh.transformations.compose_matrix(scale=list(1/self.scale)))
        self.obj.update()
"""
