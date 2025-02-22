
# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Karhan Kayan



import bpy
from infinigen.assets.utils.object import new_bbox
from infinigen.core.tagging import tag_support_surfaces
from .base import ThreedFrontFactory
import math
from infinigen_examples.util.visible import invisible_others, visible_others

def threedfront_object_factory(
    tag_support=False,
) -> ThreedFrontFactory:
    """
    Create a factory for external asset import.
    tag_support: tag the planes of the object that are parallel to xy plane as support surfaces (e.g. shelves)
    """

    class ThreedFrontCategoryFactory(ThreedFrontFactory):
        _category = None
        _asset_file = None
        _scale = None
        _rotation = None
        _position = None
       
        def __init__(self, factory_seed, coarse=False):
            super().__init__(factory_seed, coarse)
            self.tag_support = tag_support
            self.category = self._category
            self.asset_file = self._asset_file
            self.scale = self._scale
            self.rotation_orig = self._rotation
            self.location_orig = self._position
            
            
        def create_asset(self, **params) -> bpy.types.Object:
            print(self.asset_file)
            if self.asset_file == "/home/yandan/dataset/3D-scene/3D-FUTURE-model/2250c149-cae7-47b1-b2f6-061f171b3198/raw_model.obj":
                a = 1
            bpy.ops.import_scene.obj(filepath=self.asset_file)
            imported_obj = bpy.context.selected_objects[0]

            #resize
            imported_obj.scale = self.scale
            bpy.context.view_layer.objects.active = (
                imported_obj  # Set as active object
            )
            imported_obj.select_set(True)  # Select the object
            bpy.ops.object.transform_apply(
                location=False, rotation=False, scale=True
            )

            #rotate
            # uniform to front rotation
            imported_obj.rotation_mode = 'XYZ'
            radians = math.radians(90)
            # self.rotation_orig = -radians
            imported_obj.rotation_euler = [radians,0,radians]  # Rotate around Z-a to face front
            bpy.context.view_layer.objects.active = (
                imported_obj  # Set as active object
            )
            imported_obj.select_set(True)  # Select the object
            bpy.ops.object.transform_apply(
                location=False, rotation=True, scale=False
            )

            if self.tag_support:
                tag_support_surfaces(imported_obj)
            
            if imported_obj:
                return imported_obj
            else:
                raise ValueError(f"Failed to import asset: {self.asset_file}")
        
        def create_placeholder(self, **kwargs) -> bpy.types.Object:
            return new_bbox(
                -1,1,-1,1,
                0,
                2,
            )

    return ThreedFrontCategoryFactory


# Create factory instances for different categories
GeneralThreedFrontFactory = threedfront_object_factory(tag_support=True)
