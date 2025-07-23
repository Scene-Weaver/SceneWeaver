# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import bpy

from infinigen.core.placement.factory import AssetFactory


class MetaSceneFactory(AssetFactory):
    is_fragile = False
    allow_transparent = False

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        pass

    def create_asset(self, placeholder, **params) -> bpy.types.Object:
        pass
