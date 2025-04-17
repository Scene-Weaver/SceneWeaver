# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Karhan Kayan

import os
import random

import bpy

from GPT.constants import OBJATHOR_ASSETS_DIR
from GPT.objaverse_retriever import ObjathorRetriever
from GPT.retrieve import ObjectRetriever
from infinigen.assets.utils.object import new_bbox
from infinigen.core.tagging import tag_support_surfaces
from infinigen.core.util.math import FixedSeed

from .base import ObjaverseFactory
from .load_asset import load_pickled_3d_asset

global Retriever
Retriever = ObjectRetriever()


class ObjaverseCategoryFactory(ObjaverseFactory):
    _category = None
    _asset_file = None
    _scale = [1,1,1]
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
        basedir = OBJATHOR_ASSETS_DIR

        if self.asset_file is not None:
            filename = f"{basedir}/{self.asset_file}/{self.asset_file}.pkl.gz"
            imported_obj = load_pickled_3d_asset(filename)
        else: 
            object_names = Retriever.retrieve_object_by_cat(self.category)
            
            object_names_35 = [name for name, score in object_names if score > 35]
            object_names_35 = [obj_name for obj_name in object_names_35 if os.path.exists(f"{basedir}/{obj_name}")]
            if len(object_names_35)>=3:
                object_names = object_names_35
            else:
                object_names_30 = [name for name, score in object_names if score > 30]
                object_names_30 = [obj_name for obj_name in object_names_30 if os.path.exists(f"{basedir}/{obj_name}")]
                object_names = object_names_30[:3]

            import time
            random.seed(time.time())
            random.shuffle(object_names)

            for obj_name in object_names:
                filename = f"{basedir}/{obj_name}/{obj_name}.pkl.gz"
                try:
                    imported_obj = load_pickled_3d_asset(filename)
                    break
                except:
                    continue
        
        # update scale
        if (
            self.x_dim is not None
            and self.y_dim is not None
            and self.z_dim is not None
        ):
            if self.x_dim is not None:
                scale_x = self.x_dim / imported_obj.dimensions[0]
            if self.y_dim is not None:
                scale_y = self.y_dim / imported_obj.dimensions[1]
            if self.z_dim is not None:
                scale_z = self.z_dim / imported_obj.dimensions[2]
            self.scale = (scale_x, scale_y, scale_z)
        
        imported_obj.scale = self.scale
        bpy.context.view_layer.objects.active = (
            imported_obj  # Set as active object
        )
        imported_obj.select_set(True)  # Select the object
        bpy.ops.object.transform_apply(
            location=False, rotation=False, scale=True
        )
         
        if self.tag_support:
            tag_support_surfaces(imported_obj)

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
