import bpy
import mathutils
import numpy as np


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


# === User inputs ===
original_obj_name = "OldObject"  # Name of the object you want to replace
glb_path = "~/.objaverse/hf-objaverse-v1/glbs/000-007/8f2a04cbcf3741088978399924b46a92.glb"  # Path to the new .glb model

# === Find the original object ===
original_obj = bpy.data.objects.get(
    "ObjaverseCategoryFactory(9310699).spawn_asset(3669836)"
)
if original_obj is None:
    raise ValueError(f"Object named '{original_obj_name}' not found")

# === Save original transform info ===
original_loc = original_obj.location.copy()
original_rot = original_obj.rotation_euler.copy()
original_scale = original_obj.dimensions.copy()

# === Import new object ===
bpy.ops.import_scene.gltf(filepath=glb_path)
imported_objs = [obj for obj in bpy.context.selected_objects]

# === Assume top-level imported object is the root ===
new_obj = imported_objs[0]

set_origin(new_obj)


# === Compute scale factor ===
new_obj_dims = new_obj.dimensions
scale_factors = mathutils.Vector(
    (
        original_scale.x / new_obj_dims.x if new_obj_dims.x else 1,
        original_scale.y / new_obj_dims.y if new_obj_dims.y else 1,
        original_scale.z / new_obj_dims.z if new_obj_dims.z else 1,
    )
)

# Apply average scale (to preserve proportions)
avg_scale = sum(scale_factors) / 3
new_obj.scale *= avg_scale

# === Match transform ===
new_obj.location = original_loc
new_obj.rotation_euler = original_rot

# === Clean up: remove original ===
bpy.data.objects.remove(original_obj, do_unlink=True)

print(
    f"Replaced '{original_obj_name}' with imported object '{new_obj.name}' from '{glb_path}'"
)
