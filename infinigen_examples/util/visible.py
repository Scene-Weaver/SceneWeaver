from infinigen.core.util import blender as butil
import bpy

def invisible_others(hide_placeholder=False,hide_all=False):
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

    if hide_placeholder:
        mesh = butil.get_collection("placeholders")
        mesh.hide_viewport = True
        mesh.hide_render = True
    return


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
