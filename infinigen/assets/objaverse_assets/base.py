# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import random

import bpy

# from GPT.constants import OBJATHOR_ASSETS_DIR
from infinigen.assets.utils.object import new_bbox
from infinigen.core.placement.factory import AssetFactory


class ObjaverseFactory(AssetFactory):
    is_fragile = False
    allow_transparent = False

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(  # LAST
            -(self.size_extrude + 1) * self.size,  # x min
            0,  # x max
            0,  # y min
            self.width,  # y max
            -self.stand_height if self.has_stand else 0,  # z_min
            self.depth,  # z max
        )

    def create_asset(self, placeholder, **params) -> bpy.types.Object:
        from .load_asset import load_pickled_3d_asset

        cat = "book stack"
        object_names = self.retriever.retrieve_object_by_cat(cat)
        object_names = [name for name, score in object_names if score > 30]
        random.shuffle(object_names)

        for obj_name in object_names:
            basedir = OBJATHOR_ASSETS_DIR
            # indir = f"{basedir}/processed_2023_09_23_combine_scale"
            filename = f"{basedir}/{obj_name}/{obj_name}.pkl.gz"
            try:
                obj = load_pickled_3d_asset(filename)
                break
            except:
                continue

        return obj
