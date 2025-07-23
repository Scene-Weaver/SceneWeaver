# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Karhan Kayan

import json
import math
import os

import bpy

from GPT.constants import OBJATHOR_ASSETS_DIR
from infinigen.assets.utils.object import new_bbox
from infinigen.core.tagging import tag_support_surfaces

from .base import ObjaverseFactory
from .load_asset import load_pickled_3d_asset
from .place_in_blender import (
    delete_object_with_children,
    select_meshes_under_empty,
)


class ObjaverseCategoryFactory(ObjaverseFactory):
    _category = None
    _asset_file = None
    _scale = [1, 1, 1]
    _rotation = None
    _position = None
    _tag_support = True
    _x_dim = None
    _y_dim = None
    _z_dim = None

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.tag_support = self._tag_support
        self.category = self._category
        self.asset_file = self._asset_file
        self.scale = self._scale
        self.rotation_orig = self._rotation
        self.location_orig = self._position
        self.x_dim = self._x_dim
        self.y_dim = self._y_dim
        self.z_dim = self._z_dim

    def create_asset(self, **params) -> bpy.types.Object:
        if (self.asset_file is not None) and (
            not self.asset_file.endswith(".glb")
        ):  # from holodeck
            basedir = OBJATHOR_ASSETS_DIR
            filename = f"{basedir}/{self.asset_file}/{self.asset_file}.pkl.gz"
            imported_obj = load_pickled_3d_asset(filename)
        else:  # from openshape
            if self.asset_file is not None:
                filename = self.asset_file
                front_view_angle = 0
            else:
                save_dir = os.getenv("save_dir")
                with open(f"{save_dir}/objav_files.json", "r") as f:
                    LoadObjavFiles = json.load(f)
                filename = LoadObjavFiles[self.category][0]
                with open(filename.replace(".glb", "") + "/metadata.json", "r") as f:
                    value = json.load(f)["front_view"]
                    front_view_angle = value.split("/")[-1].split(".")[0].split("_")[-1]
                    angle_bias = value.split("/")[-1].split(".")[0].split("_")[1]
                    front_view_angle = int(front_view_angle) + int(angle_bias)
            bpy.ops.import_scene.gltf(filepath=filename)

            # preprocess directary
            parent_obj = bpy.context.selected_objects[0]
            # parents = get_highest_parent_objects()

            bpy.ops.object.select_all(action="DESELECT")
            select_meshes_under_empty(parent_obj.name)

            bpy.ops.object.join()
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

            joined_object = bpy.context.view_layer.objects.active
            if joined_object is not None:
                joined_object.name = parent_obj.name + "-joined"
                joined_object.location = (0, 0, 0)
                bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
                bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
                joined_object.location = (0, 0, 0)
                bpy.context.view_layer.objects.active = joined_object
                bpy.ops.object.select_all(action="DESELECT")
                joined_object.select_set(True)
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                joined_object.rotation_mode = "XYZ"
                radians = math.radians(front_view_angle + 90)
                joined_object.rotation_euler[2] = (
                    radians  # Rotate around Z-a to face front
                )
                bpy.ops.object.transform_apply(
                    location=False, rotation=True, scale=False
                )
                bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

            bpy.ops.object.select_all(action="DESELECT")
            delete_object_with_children(parent_obj)

            imported_obj = joined_object

        imported_obj.location = [0, 0, 0]
        # imported_obj.rotation_euler = [0,0,0]
        bpy.context.view_layer.objects.active = imported_obj
        bpy.ops.object.select_all(action="DESELECT")
        imported_obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)

        # update scale
        if self.x_dim is not None and self.y_dim is not None and self.z_dim is not None:
            if self.x_dim is not None:
                scale_x = self.x_dim / imported_obj.dimensions[0]
            if self.y_dim is not None:
                scale_y = self.y_dim / imported_obj.dimensions[1]
            if self.z_dim is not None:
                scale_z = self.z_dim / imported_obj.dimensions[2]
            self.scale = (scale_x, scale_y, scale_z)

        imported_obj.scale = self.scale
        bpy.context.view_layer.objects.active = imported_obj  # Set as active object
        bpy.ops.object.select_all(action="DESELECT")
        imported_obj.select_set(True)  # Select the object
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        self.set_origin(imported_obj)

        if self.tag_support:
            tag_support_surfaces(imported_obj)
        # import pdb
        # pdb.set_trace()
        print("-----------------------------------", filename)
        if imported_obj:
            return imported_obj
        else:
            raise ValueError(f"Failed to import asset: {self.asset_file}")

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -self.x_dim / 2,
            self.x_dim / 2,
            -self.y_dim / 2,
            self.y_dim / 2,
            0,
            self.z_dim,
        )


# Create factory instances for different categories
GeneralObjavFactory = ObjaverseCategoryFactory
