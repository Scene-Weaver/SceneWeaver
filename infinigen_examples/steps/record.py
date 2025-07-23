import os

import bpy

from infinigen.core.constraints.example_solver import (
    populate,
)

from .tools import export_layout, load_record, render_scene, save_record


def record_scene(
    state,
    solver,
    terrain,
    house_bbox,
    solved_bbox,
    camera_rigs,
    iter,
    p,
    transparent=False,
):
    save_dir = os.getenv("save_dir")
    export_layout(state, solver, f"{save_dir}/record_scene/layout_{iter}.json")
    p.run_stage(
        "populate_assets",
        populate.populate_state_placeholders_mid,
        state,
        # final=True,
        use_chance=False,
    )

    save_record(state, solver, terrain, house_bbox, solved_bbox, iter, p)
    save_dir = os.getenv("save_dir")
    if transparent:
        filename = f"{save_dir}/record_scene/render_{iter}.png"
    else:
        filename = f"{save_dir}/record_scene/render_{iter}.jpg"
    render_scene(
        p,
        solved_bbox,
        camera_rigs,
        state,
        solver,
        filename=filename,
        transparent=transparent,
    )

    return


def load_scene(iter):
    return load_record(iter)


def export_supporter(state, obj_name, export_path):
    def export_obj_blend(obj_name, export_path):
        populate_obj_name = state.objs[obj_name].populate_obj
        obj = bpy.data.objects.get(populate_obj_name)
        obj.location = [0, 0, 0]
        obj.rotation_euler = [0, 0, 0]
        obj.scale = [1, 1, 1]
        name = obj.name

        if obj:
            # Deselect all objects
            bpy.ops.object.select_all(action="DESELECT")
            # Select only the object you want to export
            obj.select_set(True)
            # Save only the selected object to the new .blend file
            bpy.ops.wm.save_as_mainfile(
                filepath=export_path, check_existing=False, compress=False, copy=True
            )

            # Open the new file
            bpy.ops.wm.open_mainfile(filepath=export_path)
            # Delete all objects except the selected one
            for o in bpy.data.objects:
                if o.name != name:
                    bpy.data.objects.remove(o, do_unlink=True)
            # Save only the remaining object
            bpy.ops.wm.save_as_mainfile(filepath=export_path)

            candidate = "~/workspace/SceneWeaver/record_files/obj.blend"
            bpy.ops.wm.save_as_mainfile(
                filepath=candidate, check_existing=False, compress=False, copy=True
            )

        return

    export_obj_blend(obj_name, export_path)

    return
