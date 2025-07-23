import math
import os
from math import radians

import bpy
import mathutils
from mathutils import Euler, Vector


##add light
def add_light(size=100, strength=10):
    obj = bpy.data.objects.get("newroom_0-0.floor")
    global_bbox = [
        obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box
    ]
    min_x, min_y, min_z = (min(v[i] for v in global_bbox) for i in range(3))
    max_x, max_y, max_z = (max(v[i] for v in global_bbox) for i in range(3))

    # Clear existing lights (optional)
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create area light
    light_data = bpy.data.lights.new(name="AreaLight", type="AREA")
    light_obj = bpy.data.objects.new(name="AreaLight", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)

    # Set light parameters
    light = light_obj.data
    light.color = (1.0, 0.9, 0.8)  # Warm white (RGB)
    light.shape = "RECTANGLE"  # Light shape type

    # Set size (width, height in meters)
    light.size = max_x - min_x  # Width
    light.size_y = max_y - min_y  # Height

    # Auto-calculate power based on area (scaling factor can be adjusted)
    light_area = light.size * light.size_y
    light.energy = (
        light_area * strength
    )  # 500 watts per square meter (adjust as needed)

    # Position the light
    light_obj.location = (
        (min_x + max_x) / 2,
        (min_y + max_y) / 2,
        4,
    )  # Above the scene (X,Y,Z)

    # Make it visible in renders
    #    light.cycles.use_multiple_importance_sampling = True

    print(
        f"Created {light.size}x{light.size_y}m area light with {light.energy:.0f}W power"
    )


def create_plane(
    name="MyPlane", location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)
):
    """
    Creates a plane with specified transform properties
    Args:
        name (str): Object name
        location (tuple): (x, y, z) position
        rotation (tuple): Euler angles in degrees (x, y, z)
        scale (tuple): Scale factors (x, y, z)
    """
    # Add plane mesh
    bpy.ops.mesh.primitive_plane_add(size=1)
    # plane = bpy.context.object
    plane = bpy.data.objects.get("Plane")

    # Rename
    plane.name = name
    plane.data.name = f"{name}_Mesh"

    # Set transforms
    plane.location = location
    plane.rotation_euler = (
        radians(rotation[0]),
        radians(rotation[1]),
        radians(rotation[2]),
    )
    plane.scale = scale

    # Optional: Set origin to geometry center
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")

    return plane


# add wall
def make_transparent(name, color=(0.86, 0.83, 0.79, 1)):
    # Get the object
    obj = bpy.data.objects.get(name)
    if not obj:
        raise Exception("Object 'Plane' not found")

    # Create or get material
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="OneSided_Invisible")
        obj.data.materials.append(mat)
    else:
        mat = obj.data.materials[0]

    # Enable nodes and clear existing
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Create necessary nodes
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix_shader = nodes.new("ShaderNodeMixShader")
    geometry = nodes.new("ShaderNodeNewGeometry")
    output = nodes.new("ShaderNodeOutputMaterial")

    # Position nodes for clarity
    bsdf.location = (-400, 200)
    transparent.location = (-400, -200)
    geometry.location = (-600, 0)
    mix_shader.location = (-200, 0)
    output.location = (0, 0)

    # Link nodes
    links.new(geometry.outputs["Backfacing"], mix_shader.inputs["Fac"])
    links.new(bsdf.outputs["BSDF"], mix_shader.inputs[1])  # Front face (opaque)
    links.new(
        transparent.outputs["BSDF"], mix_shader.inputs[2]
    )  # Back face (transparent)
    links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])

    # Configure material for perfect one-sided transparency
    mat.use_backface_culling = True  # Improves performance
    mat.blend_method = "CLIP"  # Sharp transparency cutoff
    mat.shadow_method = "CLIP"  # Shadows respect transparency
    mat.alpha_threshold = 0.5  # Binary visibility threshold

    # Make front face fully opaque
    bsdf.inputs["Base Color"].default_value = color  # Red for visibility
    bsdf.inputs["Alpha"].default_value = 1.0  # Front face completely opaque

    # Force refresh
    obj.active_material = mat


def add_wall(color=(0.6, 0.45, 0.2, 1)):
    # add wall
    obj = bpy.data.objects.get("newroom_0-0.floor")
    global_bbox = [
        obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box
    ]
    min_x, min_y, min_z = (min(v[i] for v in global_bbox) for i in range(3))
    max_x, max_y, max_z = (max(v[i] for v in global_bbox) for i in range(3))

    print(f"Global BBox Corners: {[min_x, min_y, min_z]} to {[max_x, max_y, max_z]}")

    # Example Usage
    name = "wall_left"  # "wall_270" #"wall_left"
    create_plane(
        name=name,
        location=((max_x - min_x) / 2 + min_x, min_y, 1.4 + min_z),
        rotation=(90, 0, 180),  # Rotated 90° on X, 45° on Z
        scale=(max_x - min_x, 2.8, 1),  # Stretched to 3m x 2m
    )
    make_transparent(name, color=color)

    # Example Usage
    name = "wall_right"  # "wall_90" #"wall_right"
    create_plane(
        name=name,
        location=((max_x - min_x) / 2 + min_x, max_y, 1.4 + min_z),
        rotation=(90, 0, 0),  # Rotated 90° on X, 45° on Z
        scale=(max_x - min_x, 2.8, 1),  # Stretched to 3m x 2m
    )
    make_transparent(name, color=color)

    # Example Usage
    name = "wall_far"  # "wall_180" #"wall_far"
    create_plane(
        name=name,
        location=(min_x, (max_y - min_y) / 2 + min_y, 1.4 + min_z),
        rotation=(90, 0, 90),  # Rotated 90° on X, 45° on Z
        scale=(max_y - min_y, 2.8, 1),  # Stretched to 3m x 2m
    )
    make_transparent(name, color=color)

    # Example Usage
    name = "wall_close"  # "wall_0" #"wall_close"
    create_plane(
        name=name,
        location=(max_x, (max_y - min_y) / 2 + min_y, 1.4 + min_z),
        rotation=(90, 0, 270),  # Rotated 90° on X, 45° on Z
        scale=(max_y - min_y, 2.8, 1),  # Stretched to 3m x 2m
    )
    make_transparent(name, color=color)


# === 1. 清理 placeholder ===
def delete_collections_and_objects_by_name_keywords(keywords):
    for coll in list(bpy.data.collections):
        if any(kw in coll.name.lower() for kw in keywords):
            # 删除集合中的所有对象
            for obj in list(coll.objects):
                coll.objects.unlink(obj)
                bpy.data.objects.remove(obj, do_unlink=True)

            bpy.data.collections.remove(coll)


# === 2. 中心移动 ===
def center_scene():
    obj = bpy.data.objects.get("newroom_0-0.floor")
    global_bbox = [
        obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box
    ]
    min_x, min_y, min_z = (min(v[i] for v in global_bbox) for i in range(3))
    max_x, max_y, max_z = (max(v[i] for v in global_bbox) for i in range(3))
    center = Vector(((max_x + min_x) / 2, (max_y + min_y) / 2, (max_z + min_z) / 2))

    mesh_objects = [
        obj for obj in bpy.data.objects if obj.type == "MESH" or obj.type == "LIGHT"
    ]
    for obj in mesh_objects:
        obj.location -= center


## === 3. 创建相机 ===
def get_scene_bounds():
    #    obj = bpy.data.objects.get("newroom_0-0.floor")
    #    global_bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    #    min_x, min_y, min_z = (min(v[i] for v in global_bbox) for i in range(3))
    #    max_x, max_y, max_z = (max(v[i] for v in global_bbox) for i in range(3))
    #    min_corner = Vector((min_x,min_y,min_z))
    #    max_corner = Vector((max_x,max_y,1.5))

    min_corner = Vector((float("inf"), float("inf"), float("inf")))
    max_corner = Vector((float("-inf"), float("-inf"), float("-inf")))
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ Vector(corner)
                min_corner = Vector(map(min, min_corner, world_corner))
                max_corner = Vector(map(max, max_corner, world_corner))
    return min_corner, max_corner


def setup_camera(margin=1.05, resolution=720):
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100

    cam = bpy.data.objects.get("Camera")
    if not cam:
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        bpy.context.collection.objects.link(cam)

    bpy.context.scene.camera = cam

    min_corner, max_corner = get_scene_bounds()
    center = (min_corner + max_corner) / 2
    size_vec = max_corner - min_corner

    width = size_vec.x * margin
    height = size_vec.z * margin
    fov_deg = 60
    fov_rad = math.radians(fov_deg)

    # 最大投影尺寸和FOV
    max_extent = max(width, height)
    distance = max_extent / (2 * math.tan(fov_rad / 2))

    # cam.location = center + Vector((0, -distance*1.5, height * 0.7))
    cam.location = center + Vector((0, -distance * 1.3, distance * 1.3 / 1.732))
    cam.location.x = 0
    cam.rotation_euler = Euler((math.radians(60), 0, 0), "XYZ")
    cam.data.lens_unit = "FOV"
    cam.data.angle = fov_rad



def update_camera(margin=1.05, resolution=720):
import bpy
resolution = 2048

bpy.context.scene.render.resolution_x = resolution
bpy.context.scene.render.resolution_y = resolution
bpy.context.scene.render.resolution_percentage = 100

cam = bpy.data.objects.get("Camera")
if not cam:
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)

bpy.context.scene.camera = cam


# === 4. 创建旋转轴空对象 ===
def setup_rotation_anchor():
    try: 
        anchor = bpy.data.objects.get("RotationAnchor")
    except:
        anchor = bpy.data.objects.new("RotationAnchor", None)
        bpy.context.collection.objects.link(anchor)

    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.parent = anchor

    return anchor

#  # === 5. 渲染不同角度的视图（30度） ===
# def render_views(anchor, angles_deg, output_dir):
#     os.makedirs(output_dir, exist_ok=True)

#     for angle in angles_deg:

#         wall = bpy.data.objects.get(f"wall_{angle}")
#         wall.hide_viewport = True
#         wall.hide_render = True
#         anchor.rotation_euler = Euler((0, 0, math.radians(angle)), 'XYZ')
#         bpy.context.view_layer.update()

#         bpy.context.scene.render.filepath = os.path.join(output_dir, f"eevee_idesign_1_view_{angle:03}.png")
#         bpy.ops.render.render(write_still=True)
#         # wall.hide_viewport = False
#         # wall.hide_render = False
#         bpy.context.view_layer.update()


def render_views(anchor, angles_deg, output_dir, render_type="eevee"):
    os.makedirs(output_dir, exist_ok=True)

    for angle in angles_deg:
        wall = bpy.data.objects.get(f"wall_{angle}")
        anchor.rotation_euler = Euler((0, 0, math.radians(angle)), "XYZ")
        bpy.context.view_layer.update()

        bpy.context.scene.render.filepath = os.path.join(
            output_dir, f"{render_type}_idesign_1_view_{angle:03}.png"
        )
        bpy.ops.render.render(write_still=True)


def rotate_scene(anchor, angle):
    anchor.rotation_euler = Euler((0, 0, math.radians(angle)), "XYZ")
    bpy.context.view_layer.update()


#        break


if __name__ == "__main__":
    import sys
    #    roomdir = "/mnt/fillipo/yandan/scenesage/record_scene/layoutgpt/restaurant/restaurant_2/"
    # outdir = "/mnt/fillipo/yandan/scenesage/record_scene/holodeck/"
    # roomtypes = os.listdir(outdir)
    # for roomtype in roomtypes:
    #     for roomname in os.listdir(f"{outdir}/{roomtype}/"):
    # roomdir = f"{outdir}/{roomtype}/{roomname}"
    # roomdir = "/mnt/fillipo/yandan/scenesage/record_scene/holodeck//waitingroom/waitingroom_0"
    # blendfile = f"{roomdir}/record_files/scene_0.blend"
    # blendfile = '/mnt/fillipo/yandan/scenesage/record_scene/holodeck//waitingroom/waitingroom_0/record_files/scene_0.blend'

    # if os.path.exists(blendfile):
    #     bpy.ops.wm.open_mainfile(filepath=blendfile, load_ui=False, use_scripts=False)

    delete_collections_and_objects_by_name_keywords(["wall", "window", "placeholder", "bbox","ceiling"])
    add_light(strength=20)
    add_wall()
    center_scene()
    setup_camera(resolution=2048)
    anchor = setup_rotation_anchor()
    bpy.context.scene.render.engine = "CYCLES"  #'CYCLES' #'BLENDER_EEVEE_NEXT'
    # print("Current render engine:", bpy.context.scene.render.engine)
    rotate_scene(anchor, angle=0)

    # ~/software/blender-4.2.0-linux-x64$ ./blender --background --python ~/workspace/infinigen/render/render_scene.py
    # ~/software/blender-4.2.0-linux-x64/blender /mnt/fillipo/yandan/scenesage/record_scene/holodeck//waitingroom/waitingroom_0/record_files/scene_0.blend --background --python ~/workspace/infinigen/render/render_scene.py
