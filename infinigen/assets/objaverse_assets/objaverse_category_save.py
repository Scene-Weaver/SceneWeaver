# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Karhan Kayan


import bpy

from GPT.retrieve import ObjectRetriever
from infinigen.assets.utils.object import new_bbox
from infinigen.core.tagging import tag_support_surfaces

from .base import ObjaverseFactory
from .retrieve_idesign import clip_model, clip_prep, get_filter_fn, preprocess, retrieve

global Retriever
Retriever = ObjectRetriever()


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
        text = preprocess("A high-poly " + self.category + ", high quality")
        device = clip_model.device
        tn = clip_prep(
            text=[text], return_tensors="pt", truncation=True, max_length=76
        ).to(device)

        enc = clip_model.get_text_features(**tn).float().cpu()
        retrieved_objs = retrieve(enc, top=10, sim_th=0.1, filter_fn=get_filter_fn())
        # retrieved_objs
        filename = retrieved_objs.values()[0]
        bpy.ops.import_scene.gltf(filepath=filename)
        imported_obj = bpy.context.selected_objects[0]
        self.set_origin(imported_obj)
        imported_obj.location = [0, 0, 0]
        imported_obj.rotation_euler = [0, 0, 0]
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
        imported_obj.select_set(True)  # Select the object
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

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
