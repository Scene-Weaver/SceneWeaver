# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Karhan Kayan


import math

import bpy

from infinigen.assets.utils.object import new_bbox
from infinigen.core.tagging import tag_support_surfaces

from .base import ThreedFrontFactory


class ThreedFrontCategoryFactory(ThreedFrontFactory):
    _category = None
    _asset_file = None
    _scale = None
    _rotation = None
    _position = None
    _tag_support = True

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.tag_support = self._tag_support
        self.category = self._category
        self.asset_file = self._asset_file
        self.scale = self._scale
        self.rotation_orig = self._rotation
        self.location_orig = self._position

    def create_asset(self, **params) -> bpy.types.Object:
        # Step 1: Keep track of existing objects
        before = set(bpy.context.scene.objects)

        # Step 2: Import the OBJ file
        bpy.ops.import_scene.obj(filepath=self.asset_file)

        # Step 3: Identify new objects added by the import
        after = set(bpy.context.scene.objects)
        new_objects = list(after - before)

        # Step 4: Filter mesh objects
        mesh_objects = [obj for obj in new_objects if obj.type == "MESH"]

        # Step 5: Join meshes if more than one was imported
        if mesh_objects:
            bpy.ops.object.select_all(action="DESELECT")
            for obj in mesh_objects:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = mesh_objects[0]
            bpy.ops.object.join()
            print("Meshes joined successfully.")
        else:
            print("No mesh objects found to join.")

        imported_obj = bpy.context.view_layer.objects.active
        # imported_obj = bpy.context.selected_objects[0]

        # bpy.ops.object.mode_set(mode='EDIT')  # Switch to Edit Mode
        # bpy.ops.mesh.select_all(action='SELECT')
        # bpy.ops.mesh.remove_doubles(threshold=1e-6)  # Merge very close vertices
        # bpy.ops.object.mode_set(mode='OBJECT')  # Sw

        # resize
        imported_obj.scale = self.scale
        bpy.context.view_layer.objects.active = imported_obj  # Set as active object
        imported_obj.select_set(True)  # Select the object
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # rotate
        # uniform to front rotation
        imported_obj.rotation_mode = "XYZ"
        radians = math.radians(90)
        # self.rotation_orig = -radians
        imported_obj.rotation_euler = [
            radians,
            0,
            radians,
        ]  # Rotate around Z-a to face front
        bpy.context.view_layer.objects.active = imported_obj  # Set as active object
        imported_obj.select_set(True)  # Select the object
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

        # imported_obj,pos_bias = self.set_origin(imported_obj)
        # self.location_orig =  [self.location_orig[i]+pos_bias[i] for i in range(3)]

        if self.tag_support:
            tag_support_surfaces(imported_obj)

        if imported_obj:
            return imported_obj
        else:
            raise ValueError(f"Failed to import asset: {self.asset_file}")

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -1,
            1,
            -1,
            1,
            0,
            2,
        )


# Create factory instances for different categories
GeneralThreedFrontFactory = ThreedFrontCategoryFactory
