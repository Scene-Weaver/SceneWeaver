# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

import bpy
import numpy as np

import GPT
from infinigen.assets.objects.seating import bedframe


class BedFactory(bedframe.BedFrameFactory):
    def __init__(self, factory_seed, coarse=False):
        super(BedFactory, self).__init__(factory_seed, coarse)
        self.retriever = GPT.Retriever

    def create_asset(self, i, **params) -> bpy.types.Object:
        placeholder = params["placeholder"]
        placeholder_size = placeholder.dimensions

        # from ..objaverse.base import load_pickled_3d_asset
        # cat = "bed with quilt"
        # object_names = self.retriever.retrieve_object_by_cat(cat)
        # import pdb
        # pdb.set_trace()
        # object_names = [name for name, score in object_names if score > 30]
        # random.shuffle(object_names)

        # for obj_name in object_names:
        #     basedir = OBJATHOR_ASSETS_DIR
        #     # indir = f"{basedir}/processed_2023_09_23_combine_scale"
        #     filename = f"{basedir}/{obj_name}/{obj_name}.pkl.gz"
        #     try:
        #         obj = load_pickled_3d_asset(filename)
        #         break
        #     except:
        #         continue

        # mesh_path = "~/dataset/3D-scene/3D-FUTURE-model/401aa29f-6dbe-41f6-b91d-9c7bf5af9dd8/raw_model.obj "
        mesh_path = "~/Desktop/bed1.obj"
        _ = bpy.ops.wm.obj_import(filepath=mesh_path)
        obj = bpy.context.selected_objects[0]
        scale = np.array(
            [
                placeholder_size[0] / obj.dimensions[0],
                placeholder_size[1] / obj.dimensions[1],
                placeholder_size[2] / obj.dimensions[2],
            ]
        )

        obj.scale = scale

        return obj
