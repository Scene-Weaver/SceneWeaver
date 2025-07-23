import bpy

from infinigen.core.util import blender as butil


def invisible_others(hide_placeholder=False, hide_all=False):
    if hide_all:
        for col in bpy.data.collections:
            if col.name == "Collection":
                continue
            mesh = butil.get_collection(col.name)
            mesh.hide_viewport = True
            mesh.hide_render = True
        return

    # rooms_split["exterior"].hide_viewport = True
    # rooms_split["exterior"].hide_render = True
    mesh = butil.get_collection("placeholders:room_shells")
    mesh.hide_viewport = True
    mesh.hide_render = True
    mesh = butil.get_collection("placeholders:portal_cutters")
    mesh.hide_viewport = True
    mesh.hide_render = True
    mesh = butil.get_collection("placeholders:room_meshes")
    mesh.hide_viewport = True
    mesh.hide_render = True
    try:
        mesh = butil.get_collection("unique_assets:room_ceiling")
        mesh.hide_viewport = True
        mesh.hide_render = True
        mesh = butil.get_collection("unique_assets:room_exterior")
        mesh.hide_viewport = True
        mesh.hide_render = True
        mesh = butil.get_collection("unique_assets:room_wall")
        mesh.hide_viewport = True
        mesh.hide_render = True
    except:
        mesh = None

    if hide_placeholder:
        mesh = butil.get_collection("placeholders")
        mesh.hide_viewport = True
        mesh.hide_render = True
    return


def invisible_wall():
    # rooms_split["exterior"].hide_viewport = True
    # rooms_split["exterior"].hide_render = True
    mesh = butil.get_collection("unique_assets:room_ceiling")
    mesh.hide_viewport = True
    mesh.hide_render = True

    mesh = butil.get_collection("unique_assets:room_wall")
    mesh.hide_viewport = True
    mesh.hide_render = True
    mesh = butil.get_collection("unique_assets:windows")
    mesh.hide_viewport = True
    mesh.hide_render = True

    return


def find_layer_collection(layer_coll, coll_name):
    # 1. 检查当前层集合是否匹配
    if layer_coll.name == coll_name:
        return layer_coll

    # 2. 递归检查所有子集合
    for child in layer_coll.children:
        found = find_layer_collection(child, coll_name)
        if found:
            return found

    # 3. 未找到返回None
    return None


def visible_layers():
    for collection in bpy.data.collections:
        collection_name = collection.name
        visible_layer(collection_name)


def visible_layer(collection_name):
    mesh = bpy.data.collections[collection_name]
    mesh.hide_viewport = False

    view_layer = bpy.context.view_layer
    layer_collection = find_layer_collection(
        view_layer.layer_collection, collection_name
    )

    if layer_collection:
        # print(f"        exclude: {layer_collection.exclude} (should be False)")
        # print(f"        hide_viewport: {layer_collection.hide_viewport} (should be False)")

        layer_collection.exclude = False
        layer_collection.hide_viewport = False
    else:
        print("   - Error: collection not found in view_layer")


def visible_others(view_all=False):
    if view_all:
        for col in bpy.data.collections:
            mesh = butil.get_collection(col.name)
            mesh.hide_viewport = False
            mesh.hide_render = False
        return

    # rooms_split["exterior"].hide_viewport = True
    # rooms_split["exterior"].hide_render = True
    mesh = butil.get_collection("placeholders:room_shells")
    mesh.hide_viewport = False
    mesh.hide_render = False
    # invisible_to_camera.apply(mesh.objects)
    mesh = butil.get_collection("placeholders:portal_cutters")
    mesh.hide_viewport = False
    mesh.hide_render = False
    # invisible_to_camera.apply(mesh.objects)
    mesh = butil.get_collection("placeholders:room_meshes")
    mesh.hide_viewport = False
    mesh.hide_render = False
    mesh = butil.get_collection("placeholders")
    mesh.hide_viewport = False
    mesh.hide_render = False
    # invisible_to_camera.apply(mesh.objects)
    return


def invisible_objects(hide_placeholder=False):
    # rooms_split["exterior"].hide_viewport = True
    # rooms_split["exterior"].hide_render = True
    mesh = butil.get_collection("placeholders:room_shells")
    mesh.hide_viewport = True
    mesh.hide_render = True
    mesh = butil.get_collection("placeholders:portal_cutters")
    mesh.hide_viewport = True
    mesh.hide_render = True
    mesh = butil.get_collection("placeholders:room_meshes")
    mesh.hide_viewport = True
    mesh.hide_render = True
    if hide_placeholder:
        mesh = butil.get_collection("placeholders")
        mesh.hide_viewport = True
        mesh.hide_render = True
    return
