import math
import os
import sys

import bpy
from mathutils import Vector

# sys.path.append("~/workspace/Tabletop-Digital-Cousins/digital_cousins/models/blend/") #TODO
import mathutils
import numpy as np
from scipy.spatial.transform import Rotation


def set_origin(imported_obj):
    imported_obj.location = [0, 0, 0]
    bbox_corners = [mathutils.Vector(corner) for corner in imported_obj.bound_box]

    min_z = min(corner.z for corner in bbox_corners)
    imported_obj.location.z -= min_z

    mean_x = np.mean([corner.x for corner in bbox_corners])
    imported_obj.location.x -= mean_x
    mean_y = np.mean([corner.y for corner in bbox_corners])
    imported_obj.location.y -= mean_y

    pos_bias = [mean_x, mean_y, min_z]
    bpy.context.scene.cursor.location = [0, 0, 0]

    bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="BOUNDS")
    return imported_obj, pos_bias


def add_light():
    for obj in bpy.data.objects:
        if obj.name.startswith("Area"):
            return
            # bpy.data.objects.remove(obj)

    # Create key light (main light)
    bpy.ops.object.light_add(type="AREA", location=(4, 4, 5))
    # key_light = bpy.context.object
    key_light = bpy.context.view_layer.objects.active
    key_light.data.energy = 1000  # Adjust intensity
    key_light.data.size = 5  # Softness of the shadows

    # Create fill light (soft light)
    bpy.ops.object.light_add(type="AREA", location=(-4, -4, 5))
    fill_light = bpy.context.view_layer.objects.active
    fill_light.data.energy = 500  # Lower intensity than key light
    fill_light.data.size = 5  # Soft shadows

    # Create back light (rim light)
    bpy.ops.object.light_add(type="AREA", location=(0, 5, 5))
    back_light = bpy.context.view_layer.objects.active
    back_light.data.energy = 300  # Low intensity
    back_light.data.size = 3  # Soft shadows

    # Set the world background to an HDRi for ambient lighting
    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    bg_node = world_nodes["Background"]
    bg_node.inputs["Color"].default_value = (
        0.05,
        0.05,
        0.05,
        1,
    )  # Very subtle background color for studio effect

    return


def new_angle_range(top_k_angles):
    top_k_angles.sort()
    angle1, angle2 = top_k_angles
    if angle2 - angle1 > angle1 + 360 - angle2:
        start_angle = angle2
        end_angle = angle1 + 360
    else:
        start_angle = angle1
        end_angle = angle2
    assert start_angle <= end_angle
    return start_angle, end_angle


def render_rotation(
    obj,
    save_dir,
    start_angle,
    end_angle,
    cnt,
    rot_z=0,
    rot_axis=0,
    angle_axis=0,
    z_angle=0,
    background="1",
):
    if background == "0":
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    # set initial pose
    obj.rotation_euler = (0, 0, 0)
    radians = math.radians(rot_z)
    obj.rotation_euler[2] = radians  # Rotate around Z-a
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(rotation=True)
    obj.select_set(False)
    if background == "0":
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    if rot_axis == 2:
        angle_axis = 0
    obj.rotation_euler[rot_axis] = math.radians(angle_axis)
    bpy.context.view_layer.objects.active = obj
    # obj.select_set(True)
    obj.select_set(True)
    bpy.ops.object.transform_apply(rotation=True)
    obj.select_set(False)
    if background == "0":
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)

    os.system(f"rm {save_dir}/*")
    scene = bpy.context.scene

    try:
        # Delete the cube
        cube = bpy.data.objects["Cube"]
        bpy.context.view_layer.objects.active = cube  # Set the cube as active
        bpy.ops.object.delete()
    except:
        pass

    # Run the function
    position_camera_to_fit()
    # camera = scene.camera
    # camera.location = Vector((-0.019525, 0.7789, 0.223973))  # x, y, z coordinates
    # camera.rotation_euler = Euler((73.3269/180*math.pi, 0, 180.97/180*math.pi), 'XYZ')  # Rotation in radians
    # camera.scale = Vector((1.0, 1.0, 1.0))  # x, y, z scaling factors
    bpy.context.view_layer.update()
    scene.render.film_transparent = True
    add_light()

    # import_obj = bpy.ops.wm.obj_import(filepath = mesh_path)
    # obj = bpy.context.selected_objects[0]
    obj.rotation_euler = (0, 0, 0)

    cnt = min(end_angle - start_angle + 1, cnt)
    for i in range(cnt):
        angle = start_angle + i * (end_angle - start_angle) * 1.0 / (cnt - 1)

        # Convert degrees to radians for rotation
        radians = math.radians(angle)
        obj.rotation_euler[2] = radians  # Rotate around Z-axis

        # Optional: Update the scene to see the rotation
        bpy.context.view_layer.update()

        # Set the render settings
        bpy.context.scene.render.filepath = (
            f"{save_dir}/{rot_z}_{rot_axis}_{angle_axis}_{int(angle)}.png"
        )
        # bpy.context.scene.render.filepath = f"{save_dir}/{int(angle)}.png"  # Change the filepath as needed
        bpy.context.scene.render.image_settings.file_format = (
            "PNG"  # Set the desired image format
        )

        # Render the image
        bpy.ops.render.render(write_still=True)

        if background == "0":
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)

    #    obj.location = (0,0,0)
    #    obj.rotation_euler = (0,0,0)
    bpy.context.view_layer.update()
    return


def get_minimal_bounding_box(mesh, rot_z):
    # Rotate mesh around the Z-axis
    rotation_matrix = Rotation.from_euler("z", rot_z, degrees=True).as_matrix()
    rotated_vertices = mesh.vertices @ rotation_matrix.T

    # Compute axis-aligned bounding box of the rotated mesh
    min_corner = rotated_vertices.min(axis=0)
    max_corner = rotated_vertices.max(axis=0)

    # Dimensions of the bounding box
    dimensions = max_corner - min_corner

    return dimensions


def calc_minimal_bbox(mesh):
    # mesh.affine_transform(t=-self.position)
    # show(renderables, behaviours=behaviours,camera_position=(2, 2, 2),background=(0,)*4)

    for rot_z in range(46):
        dimensions = get_minimal_bounding_box(mesh, rot_z)
        area = dimensions[0] * dimensions[1]
        if rot_z == 0:
            min_area = area
            min_rot_z = rot_z
            size = dimensions / 2
            # draw_box_label([0,0,0], size, -min_rot_z, mesh, names = None)

        elif area < min_area:
            min_area = area
            min_rot_z = rot_z
            size = dimensions / 2

    return min_rot_z, size


def convert_blender_obj_to_trimesh(obj):
    import numpy as np
    import trimesh

    mesh = obj.data
    # 提取顶点坐标
    vertices = np.array([v.co for v in mesh.vertices])
    # 提取面索引
    faces = np.array([face.vertices for face in mesh.polygons])
    # 创建 Trimesh 对象
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def render_90degree(obj, save_dir):
    obj.rotation_mode = "XYZ"

    obj.rotation_euler = (0, 0, 0)
    mesh = convert_blender_obj_to_trimesh(obj)
    rot_z, size = calc_minimal_bbox(mesh)
    angle = rot_z
    radians = math.radians(angle)
    obj.rotation_euler[2] = radians  # Rotate around Z-a
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(rotation=True)
    obj.select_set(False)

    os.system(f"rm {save_dir}/*")
    scene = bpy.context.scene
    try:
        # Delete the cube
        cube = bpy.data.objects["Cube"]
        bpy.context.view_layer.objects.active = cube  # Set the cube as active
        cube.select_set(True)  # Select the cube
        bpy.ops.object.delete()
    except:
        pass

    # Run the function
    position_camera_to_fit()

    bpy.context.view_layer.update()
    scene.render.film_transparent = True
    add_light()

    bpy.context.view_layer.objects.active = obj
    for z_angle in [0, 90, 180, 270]:
        obj.rotation_euler[2] = math.radians(z_angle)
        # Optional: Update the scene to see the rotation
        bpy.context.view_layer.update()

        # Set the render settings
        bpy.context.scene.render.filepath = f"{save_dir}/rotz_{rot_z}_z_angle_{z_angle}.png"  # Change the filepath as needed
        bpy.context.scene.render.image_settings.file_format = (
            "PNG"  # Set the desired image format
        )

        # Render the image
        bpy.ops.render.render(write_still=True)

    #    obj.location = (0,0,0)
    #    obj.rotation_euler = (0,0,0)
    bpy.context.view_layer.update()
    # obj=bpy.context.view_layer.objects.active
    # obj.select_set(True)
    # bpy.ops.object.delete()
    obj = (
        bpy.context.view_layer.objects.active
    )  # Get the active object, or specify the object

    # Remove the object from the scene
    bpy.data.objects.remove(obj, do_unlink=True)

    return


# Calculate the bounding box center and size
def get_scene_bounds():
    min_coords = Vector((float("inf"), float("inf"), float("inf")))
    max_coords = Vector((float("-inf"), float("-inf"), float("-inf")))

    # Iterate through all objects and find their min/max bounds
    for obj in bpy.context.visible_objects:
        if obj.type == "MESH":
            for vertex in obj.bound_box:
                # Convert local bounding box to world coordinates
                world_vertex = obj.matrix_world @ Vector(vertex)
                min_coords = Vector(
                    (min(min_coords[i], world_vertex[i]) for i in range(3))
                )
                max_coords = Vector(
                    (max(max_coords[i], world_vertex[i]) for i in range(3))
                )

    center = (max_coords + min_coords) / 2
    size = max_coords - min_coords
    return center, size


# Position the camera to fit the bounding box
def position_camera_to_fit():
    center, size = get_scene_bounds()
    scene_radius = max(size / 2)

    # Select or create the camera
    if not bpy.context.scene.camera:
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        bpy.context.scene.camera = camera
    camera = bpy.context.scene.camera

    # Set camera rotation to 60 degrees on the X-axis
    camera.rotation_euler = (math.radians(60), 0, 0)

    # Calculate the required distance to fit the objects based on the 60-degree angle
    distance = (size.y / 2) / math.tan(
        math.radians(30)
    )  # Adjust distance based on the field of view

    # Set camera position using calculated distance
    camera.location = center + Vector((0, -5 * scene_radius, 3 * scene_radius))

    # Adjust focal length if necessary for a wider field of view
    camera.data.lens = 35  # You can modify this value if you need a wider/narrower view

    # Ensure camera is set as the active camera
    bpy.context.scene.camera = camera

    # Update the view
    bpy.context.view_layer.update()


def load_obj_from_blend(filename):
    bpy.ops.wm.open_mainfile(filepath=filename)

    # Get the list of objects in the scene
    objects = bpy.context.scene.objects

    # Check if there is only one object
    assert len(objects) == 1

    # Get the only object in the scene
    obj = objects[0]
    # Select the object
    obj.select_set(True)
    # Optionally, make it the active object
    bpy.context.view_layer.objects.active = obj

    return obj


def merge_obj_from_blend(filename):
    # 加载对象到当前场景
    with bpy.data.libraries.load(filename, link=False) as (data_from, data_to):
        # Ensure the blend file has at least one object
        if len(data_from.objects) == 0:
            raise RuntimeError(f"No objects found in {filename}")

        # Get the first object from the blend file
        object_name = data_from.objects[0]
        data_to.objects = [object_name]

    # Link the object into the current scene
    appended_object = data_to.objects[0]
    bpy.context.scene.collection.objects.link(appended_object)
    return appended_object


def load_glb(mesh_path):
    bpy.ops.import_scene.gltf(filepath=mesh_path)
    imported_obj = bpy.context.selected_objects[0]
    imported_obj.location = [0, 0, 0]
    # rotation = imported_obj.rotation_euler
    # scale = imported_obj.scale
    bpy.context.view_layer.objects.active = imported_obj  # Set as active object
    imported_obj.select_set(True)  # Select the object
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return imported_obj


if __name__ == "__main__":
    basedir = "/mnt/fillipo/huangyue/recon_sim/7_anno_v4/export_stage2_sm"
    for scene_name in os.listdir(basedir):
        scene_name = "scene0157_00"
        scene_path = f"{basedir}/{scene_name}"

        outdir = f"/mnt/fillipo/yandan/metascene/export_stage2_sm/{scene_name}"
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        else:
            continue

        bpy.ops.object.select_all(action="SELECT")
        # Delete all selected objects
        bpy.ops.object.delete()

        bpy.context.scene.render.resolution_x = 512  # Width in pixels
        bpy.context.scene.render.resolution_y = 512

        for i in os.listdir(scene_path):
            if i.startswith("planes") or i.startswith("geometry"):
                continue
            if i.endswith(".glb"):
                obj_name = i.split(".")[0]
                mesh_path = f"{scene_path}/{obj_name}.glb"
                imported_obj = load_glb(mesh_path)
                imported_obj, pos_bias = set_origin(imported_obj)

                save_dir = f"{outdir}/{obj_name}"
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                render_90degree(imported_obj, save_dir)

        break
