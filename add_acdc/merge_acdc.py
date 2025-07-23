import json
import math
import random

import bpy
from mathutils import Matrix

basic_object = {
    "active": True,
    "contour": None,
    "dof_matrix_translation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    "dof_rotation_axis": [-0.0, 0.0, 1.0],
    "generator": "ACDC",
    "obj": "MonitorFactory(3610974).spawn_asset(964244)",
    "relations": [
        {
            "child_plane_idx": 0,
            "parent_plane_idx": 0,
            "relation": {
                "check_z": True,
                "child_tags": [
                    "-Subpart(back)",
                    "-Subpart(top)",
                    "Subpart(bottom)",
                    "-Subpart(front)",
                ],
                "margin": 0,
                "parent_tags": [
                    "Subpart(top)",
                    "-Subpart(back)",
                    "-Subpart(front)",
                    "-Subpart(bottom)",
                ],
                "relation_type": "StableAgainst",
                "rev_normal": False,
            },
            "target_name": None,
        }
    ],
    "tags": [
        "-Semantics(room)",
        "FromGenerator(ACDC)",
        "Semantics(object)",
        "Semantics(real-placeholder)",
    ],
}


def load_infinigen_scene(scene_name):
    blend_file_path = (
        f"outputs/indoors/{scene_name}/obj.blend"
    )
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)
    bpy.context.view_layer.update()
    return



def load_acdc_scene(blend_file_path):
    collection_acdc = bpy.data.collections.new("CollectionACDC")
    bpy.context.scene.collection.children.link(collection_acdc)

    # 加载对象到当前场景
    with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
        data_to.objects = []

        for obj in data_from.objects:
            if obj.startswith("Area") or obj.startswith("Camera"):
                continue
            data_to.objects.append(obj)

    # 将加载的集合添加到当前场景
    for obj in data_to.objects:
        collection_acdc.objects.link(obj)

    bpy.context.view_layer.update()
    return


def get_obj_from_collection(collection_name, obj_name):
    collection = bpy.data.collections[collection_name]
    if obj_name in collection.objects:
        return collection.objects[obj_name]


def remove_scaling(matrix_world, rot=180):
    # Extract translation (last column)
    translation = matrix_world.to_translation()

    # Extract rotation and remove scaling by normalizing columns
    rotation_matrix = matrix_world.to_3x3()
    rotation_matrix.normalize()

    # add angle
    euler_angles = rotation_matrix.to_euler("XYZ")
    euler_angles.z += math.radians(rot)
    # Convert the adjusted Euler angles back to a rotation matrix
    rotation_matrix = euler_angles.to_matrix()

    # Reconstruct a 4x4 matrix with translation and normalized rotation
    new_matrix_world = Matrix.Translation(translation) @ rotation_matrix.to_4x4()

    # Assign the modified matrix back to the object
    return new_matrix_world


def transform_acdc(
    source_name="desk_0",
    target_name="SimpleDeskFactory(8569017).spawn_asset(5056988)",
    additional_rot=0,
):
    source_obj = get_obj_from_collection("CollectionACDC", source_name)

    target_obj = get_obj_from_collection("unique_assets", target_name)

    M_source = source_obj.matrix_world.copy()
    M_target = target_obj.matrix_world.copy()

    M_source_noscale = remove_scaling(M_source, additional_rot)

    collection = bpy.data.collections["CollectionACDC"]
    # 遍历集合中的对象
    for obj in collection.objects:
        # 确保对象不是隐藏的
        if obj.hide_viewport:
            continue
        if obj.name == source_name:
            bpy.data.objects.remove(obj)
            continue
        obj.matrix_world = M_target @ M_source_noscale.inverted() @ obj.matrix_world
        print(f"Transformed object: {obj.name}")

    return target_obj


def remove_children(target_name, scene_name):
    json_path = f"outputs/indoors/{scene_name}/solve_state.json"
    with open(json_path, "r") as f:
        j = json.load(f)

    for objname in j["objs"].keys():
        objinfo = j["objs"][objname]
        if objinfo["obj"] == target_name:
            key = objname
            break

    todelect = []
    for objname in j["objs"].keys():
        objinfo = j["objs"][objname]
        for relation in objinfo["relations"]:
            if (
                relation["target_name"] == key
                and "Subpart(bottom)" in relation["relation"]["child_tags"]
            ):
                delete_obj = get_obj_from_collection("unique_assets", objinfo["obj"])
                bpy.data.objects.remove(delete_obj)
                todelect.append(objname)

                break

    for objname in todelect:
        del j["objs"][objname]

    json_path_new = f"outputs/indoors/{scene_name}/solve_state_acdc.json"
    with open(json_path_new, "w") as f:
        json.dump(j, f, indent=4)

    return key


def update_solve_state(target_key):
    json_path_new = f"~/outputs/indoors/{scene_name}/solve_state_acdc.json"
    with open(json_path_new, "r") as f:
        j = json.load(f)

    collection = bpy.data.collections["CollectionACDC"]
    num_lst = []
    for object in collection.objects:
        num = random.randint(100000, 999999)
        while num in num_lst:
            num = random.randint(100000, 999999)
        num_lst.append(num)

        object_key = str(num) + "_" + object.name.split("_")[0]
        objinfo = basic_object.copy()
        objinfo["relations"][0]["target_name"] = target_key
        objinfo["obj"] = object.name
        j["objs"][object_key] = objinfo

    with open(json_path_new, "w") as f:
        json.dump(j, f, indent=4)
    return


def save_blend_file(scene_name):
    blend_file_path = (
        f"outputs/indoors/{scene_name}/scene_new.blend"
    )
    bpy.ops.wm.save_as_mainfile(filepath=blend_file_path)

    return


if __name__ == "__main__":
    global scene_name
    scene_idx = 1

    scene_name = "debug_book"
    acdc_file_path = "~/Desktop/acdc_objaverse/acdc_output-desk6/step_3_output/scene_1/scene_1.blend"
    target_name = "SimpleDeskFactory(1487939).spawn_asset(1733853)"
    additional_rot = 0

    load_infinigen_scene(scene_name)
    load_acdc_scene(acdc_file_path)

    source_name = "desk_0"
    target_key = remove_children(target_name, scene_name)
    target_obj = transform_acdc(source_name, target_name, additional_rot)
    update_solve_state(target_key)

    # bpy.data.objects.remove(target_obj)

    save_blend_file(scene_name)
