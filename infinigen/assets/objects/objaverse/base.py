# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import gzip
import pickle

import bpy
import numpy as np
from mathutils import Vector
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.utils.decorate import read_co, write_attribute
from infinigen.assets.utils.misc import assign_material
from infinigen.assets.utils.object import join_objects, new_bbox, origin2lowest
from infinigen.core import surface
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


class ObjaverseFactory(AssetFactory):
    is_fragile = False
    allow_transparent = False

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        # with FixedSeed(factory_seed):
        #     self.thickness = 0.01
        #     material_assignments = AssetList["TablewareFactory"](
        #         fragile=self.is_fragile, transparent=self.allow_transparent
        #     )

        #     self.surface = material_assignments["surface"].assign_material()
        #     self.inside_surface = material_assignments["inside"].assign_material()
        #     self.guard_surface = material_assignments["guard"].assign_material()

        #     scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
        #     self.scratch, self.edge_wear = material_assignments["wear_tear"]

        #     self.scratch = None if uniform() > scratch_prob else self.scratch
        #     self.edge_wear = None if uniform() > edge_wear_prob else self.edge_wear

        #     self.guard_depth = self.thickness
        #     self.has_guard = False
        #     self.has_inside = False
        #     self.lower_thresh = uniform(0.5, 0.8)
        #     self.scale = 1.0
        #     self.metal_color = "bw+natural"

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -self.side_size,
            self.side_size,
            -self.side_size,
            self.side_size,
            -0.02,
            self.base_factory.depth * self.base_factory.scale + self.top_size,
        )

    def solidify_with_inside(self, obj, thickness):
        # 计算给定对象 (obj) 的所有顶点的 Z 坐标的最大值 (z 轴最大值)
        max_z = np.max(read_co(obj)[:, -1])
        # 创建一个新的顶点组，名为 "inside_"
        obj.vertex_groups.new(name="inside_")
        # 调用 butil.modify_mesh 方法，给 obj 应用 "SOLIDIFY" 修改器，设置厚度为 thickness，偏移量为 1，内部顶点组使用 "inside_"
        butil.modify_mesh(
            obj, "SOLIDIFY", thickness=thickness, offset=1, shell_vertex_group="inside_"
        )
        # 使用 write_attribute 方法，将 "inside_" 顶点组的属性写入 "inside" 面属性中
        write_attribute(obj, "inside_", "inside", "FACE")

        def inside(nw: NodeWrangler):
            lower = nw.compare(
                "LESS_THAN",
                nw.separate(nw.new_node(Nodes.InputPosition))[-1],
                max_z * self.lower_thresh,
            )
            inside = nw.compare(
                "GREATER_THAN", surface.eval_argument(nw, "inside"), 0.8
            )
            return nw.boolean_math("AND", inside, lower)

        write_attribute(obj, inside, "lower_inside", "FACE")
        obj.vertex_groups.remove(obj.vertex_groups["inside_"])

    def create_asset(self, **params) -> bpy.types.Object:
        obj = load_pickled_3d_asset(filepath="")
        return obj


def load_pickled_3d_asset(file_path, idx=0):
    # Open the compressed pickled file
    with gzip.open(file_path, "rb") as f:
        # Load the pickled object
        loaded_object_data = pickle.load(f)

    # Create a new mesh object in Blender
    mesh = bpy.data.meshes.new(name="LoadedMesh")
    obj = bpy.data.objects.new("LoadedObject", mesh)

    # Link the object to the scene
    bpy.context.scene.collection.objects.link(obj)

    # Set the mesh data for the object
    obj.data = mesh

    # Update the mesh with the loaded data
    # print(loaded_object_data.keys())
    # print(loaded_object_data['triangles'])
    # triangles = [vertex_index for face in loaded_object_data['triangles'] for vertex_index in face]
    triangles = np.array(loaded_object_data["triangles"]).reshape(-1, 3)
    vertices = []

    for v in loaded_object_data["vertices"]:
        vertices.append([v["x"], v["z"], v["y"]])

    mesh.from_pydata(vertices, [], triangles)

    uvs = []
    for uv in loaded_object_data["uvs"]:
        uvs.append([uv["x"], uv["y"]])

    mesh.update()

    # Ensure UV coordinates are stored
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")

    uv_layer = mesh.uv_layers["UVMap"]
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            uv_layer.data[loop_index].uv = uvs[vertex_index]

    material = bpy.data.materials.new(name="AlbedoMaterial")
    obj.data.materials.append(material)

    # Assign albedo color to the material
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")

    texture_node = nodes.new(type="ShaderNodeTexImage")

    image_path = f"{'/'.join(file_path.split('/')[:-1])}/albedo.jpg"  # Replace with your image file path

    image = bpy.data.images.load(image_path)

    # Assign the image to the texture node
    texture_node.image = image

    # Connect the texture node to the albedo color
    material.node_tree.links.new(
        texture_node.outputs["Color"], principled_bsdf.inputs["Base Color"]
    )

    # normal
    image_path = f"{'/'.join(file_path.split('/')[:-1])}/normal.jpg"
    img_normal = bpy.data.images.load(image_path)
    image_texture_node_normal = material.node_tree.nodes.new(type="ShaderNodeTexImage")
    image_texture_node_normal.image = img_normal
    image_texture_node_normal.image.colorspace_settings.name = "Non-Color"

    normal_map_node = material.node_tree.nodes.new(type="ShaderNodeNormalMap")

    material.node_tree.links.new(
        normal_map_node.outputs["Normal"], principled_bsdf.inputs["Normal"]
    )
    material.node_tree.links.new(
        image_texture_node_normal.outputs["Color"], normal_map_node.inputs["Color"]
    )

    # Assign the material to the object
    obj.data.materials[0] = material

    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

    xx = [v[0] for v in bbox]
    yy = [v[1] for v in bbox]
    zz = [v[2] for v in bbox]

    length = max(xx) - min(xx)
    width = max(yy) - min(yy)
    height = max(zz) - min(zz)

    obj.location = [-(max(xx) + min(xx)) / 2, -(max(xx) + min(xx)) / 2, -min(zz)]
    # obj.location = [0,0,-height/2]
    with butil.SelectObjects(obj):
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    #    i = idx//10
    #    j = idx%10
    #    obj.location =    [0.2*i ,0.5*j, 0  ]
    import pdb

    pdb.set_trace()
    # Update mesh to apply UV changes
    mesh.update()

    return obj
