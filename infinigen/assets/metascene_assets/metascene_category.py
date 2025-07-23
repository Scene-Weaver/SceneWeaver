# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Karhan Kayan


import math

import bpy
from mathutils import Vector

from infinigen.assets.utils.object import new_bbox
from infinigen.core.tagging import tag_support_surfaces
from infinigen.core.util import blender as butil

from .base import MetaSceneFactory


def modify_obj_center(obj):
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

    xx = [v[0] for v in bbox]
    yy = [v[1] for v in bbox]
    zz = [v[2] for v in bbox]

    length = max(xx) - min(xx)
    width = max(yy) - min(yy)
    height = max(zz) - min(zz)

    bottom_bias = [(max(xx) + min(xx)) / 2, (max(xx) + min(xx)) / 2, min(zz)]

    obj.location = -bottom_bias
    # obj.location = [0,0,-height/2]
    with butil.SelectObjects(obj):
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    return obj, bottom_bias


class MetaCategoryFactory(MetaSceneFactory):
    _category = None
    _asset_file = None
    _front_view_angle = None
    _tag_support = True

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        self.tag_support = self._tag_support
        self.category = self._category
        self.asset_file = self._asset_file
        self.front_view_angle = self._front_view_angle

    def create_asset(self, **params) -> bpy.types.Object:
        print(self.asset_file)
        bpy.ops.import_scene.gltf(filepath=self.asset_file)
        imported_obj = bpy.context.selected_objects[0]

        # scale
        self.scale = list(imported_obj.scale)
        bpy.context.view_layer.objects.active = imported_obj  # Set as active object
        imported_obj.select_set(True)  # Select the object
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        # location
        self.location_orig = list(imported_obj.location.copy())
        imported_obj, pos_bias = self.set_origin(imported_obj)
        self.location_orig = [self.location_orig[i] + pos_bias[i] for i in range(3)]
        # rotation
        # uniform to front rotation
        imported_obj.rotation_mode = "XYZ"
        radians = math.radians(self.front_view_angle + 90)
        self.rotation_orig = -radians
        imported_obj.rotation_euler[2] = radians  # Rotate around Z-a to face front

        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

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

    # def set_origin(self,imported_obj):
    #     imported_obj.location = [0,0,0]
    #     bbox_corners = [mathutils.Vector(corner) for corner in imported_obj.bound_box]

    #     min_z = min(corner.z for corner in bbox_corners)
    #     imported_obj.location.z -= min_z

    #     mean_x = np.mean([corner.x for corner in bbox_corners])
    #     imported_obj.location.x -= mean_x
    #     mean_y = np.mean([corner.y for corner in bbox_corners])
    #     imported_obj.location.x -= mean_y

    #     pos_bias = [mean_x,mean_y,min_z]

    #     bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='BOUNDS')
    #     return imported_obj,pos_bias


# Create factory instances for different categories
GeneralMetaFactory = MetaCategoryFactory
