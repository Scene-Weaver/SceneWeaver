import os

import bpy


def pack_material(blend_path):
    bpy.context.preferences.filepaths.save_version = 0

    bpy.ops.wm.open_mainfile(filepath=blend_path, load_ui=False, use_scripts=False)
    bpy.ops.file.make_paths_absolute()
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, check_existing=False)
    return


if __name__ == "__main__":
    method = "atiss"
    basedir = f"/mnt/fillipo/yandan/scenesage/record_scene/{method}"
    for scene_path in os.listdir(basedir):
        blend_path = f"{basedir}/{scene_path}/record_files/scene_0.blend"
        pack_material(blend_path)

        a = 1
