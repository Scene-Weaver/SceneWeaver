import math
import os
import sys

import bpy
from mathutils import Euler, Vector


def add_light():
    # Create key light (main light)
    bpy.ops.object.light_add(type="AREA", location=(4, 4, 5))
    key_light = bpy.context.object
    key_light.data.energy = 1000  # Adjust intensity
    key_light.data.size = 5  # Softness of the shadows

    # Create fill light (soft light)
    bpy.ops.object.light_add(type="AREA", location=(-4, -4, 5))
    fill_light = bpy.context.object
    fill_light.data.energy = 500  # Lower intensity than key light
    fill_light.data.size = 5  # Soft shadows

    # Create back light (rim light)
    bpy.ops.object.light_add(type="AREA", location=(0, 5, 5))
    back_light = bpy.context.object
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


def render_rotation(mesh_path, start_angle, end_angle):
    save_dir = "~/Desktop/dift/source"

    os.system(f"rm {save_dir}/*")
    scene = bpy.context.scene

    # Delete the cube
    cube = bpy.data.objects["Cube"]
    bpy.context.view_layer.objects.active = cube  # Set the cube as active
    cube.select_set(True)  # Select the cube
    bpy.ops.object.delete()

    camera = scene.camera
    camera.location = Vector((-0.019525, 0.7789, 0.223973))  # x, y, z coordinates
    camera.rotation_euler = Euler(
        (73.3269 / 180 * math.pi, 0, 180.97 / 180 * math.pi), "XYZ"
    )  # Rotation in radians
    camera.scale = Vector((1.0, 1.0, 1.0))  # x, y, z scaling factors
    bpy.context.view_layer.update()
    scene.render.film_transparent = True
    add_light()

    import_obj = bpy.ops.wm.obj_import(filepath=mesh_path)
    obj = bpy.context.selected_objects[0]
    obj.rotation_euler = (0, 0, 0)

    cnt = min(end_angle - start_angle + 1, 20)
    for i in range(cnt):
        angle = start_angle + i * (end_angle - start_angle) * 1.0 / (cnt - 1)

        # Convert degrees to radians for rotation
        radians = math.radians(angle)
        obj.rotation_euler[2] = radians  # Rotate around Z-axis

        # Optional: Update the scene to see the rotation
        bpy.context.view_layer.update()

        # Set the render settings
        bpy.context.scene.render.filepath = (
            f"{save_dir}/{int(angle)}.png"  # Change the filepath as needed
        )
        bpy.context.scene.render.image_settings.file_format = (
            "PNG"  # Set the desired image format
        )

        # Render the image
        bpy.ops.render.render(write_still=True)

    #    obj.location = (0,0,0)
    #    obj.rotation_euler = (0,0,0)
    bpy.context.view_layer.update()
    return


if __name__ == "__main__":
    start_angle = int(sys.argv[-2])
    end_angle = int(sys.argv[-1])

    render_rotation("~/Desktop/dift/source.obj", start_angle, end_angle)
