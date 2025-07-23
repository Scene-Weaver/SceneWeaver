import bpy
import numpy as np
from numpy import deg2rad

from infinigen import repo_root
from infinigen.assets.materials import invisible_to_camera
from infinigen.assets.objects.wall_decorations.skirting_board import make_skirting_board
from infinigen.assets.placement.floating_objects import FloatingObjectPlacement
from infinigen.assets.utils.decorate import read_co
from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.example_solver.room import decorate as room_dec
from infinigen.core.constraints.example_solver.room.constants import WALL_HEIGHT
from infinigen.core.placement import camera as cam_util
from infinigen.core.util import blender as butil
from infinigen.core.util.camera import points_inview
from infinigen.core.util.test_utils import (
    import_item,
    load_txt_list,
)
from infinigen_examples.util.generate_indoors_util import (
    create_outdoor_backdrop,
    hide_other_rooms,
)


def finalize_scene(
    overrides,
    stages,
    state,
    solver,
    output_folder,
    p,
    terrain,
    solved_rooms,
    house_bbox,
    camera_rigs,
):
    def place_floating():
        pholder_rooms = butil.get_collection("placeholders:room_meshes")
        pholder_cutters = butil.get_collection("placeholders:portal_cutters")
        pholder_objs = butil.get_collection("placeholders")

        obj_fac_names = load_txt_list(
            repo_root() / "tests" / "assets" / "list_indoor_meshes.txt"
        )
        facs = [import_item(path) for path in obj_fac_names]

        placer = FloatingObjectPlacement(
            generators=facs,
            camera=cam_util.get_camera(0, 0),
            background_objs=list(pholder_cutters.objects) + list(pholder_rooms.objects),
            collision_objs=list(pholder_objs.objects),
        )

        placer.place_objs(
            num_objs=overrides.get("num_floating", 20),
            normalize=overrides.get("norm_floating_size", True),
            collision_placed=overrides.get("enable_collision_floating", False),
            collision_existing=overrides.get("enable_collision_solved", False),
        )

    p.run_stage("floating_objs", place_floating, use_chance=False, default=state)
    # endregion

    # region final step
    door_filter = r.Domain({t.Semantics.Door}, [(cl.AnyRelation(), stages["rooms"])])
    window_filter = r.Domain(
        {t.Semantics.Window}, [(cl.AnyRelation(), stages["rooms"])]
    )
    p.run_stage(
        "room_doors",
        lambda: room_dec.populate_doors(solver.get_bpy_objects(door_filter)),
        use_chance=False,
    )

    p.run_stage(
        "room_windows",
        lambda: room_dec.populate_windows(solver.get_bpy_objects(window_filter)),
        use_chance=False,
    )

    room_meshes = solver.get_bpy_objects(r.Domain({t.Semantics.Room}))
    p.run_stage(
        "room_stairs",
        lambda: room_dec.room_stairs(state, room_meshes),
        use_chance=False,
    )
    p.run_stage(
        "skirting_floor",
        lambda: make_skirting_board(room_meshes, t.Subpart.SupportSurface),
    )
    p.run_stage(
        "skirting_ceiling", lambda: make_skirting_board(room_meshes, t.Subpart.Ceiling)
    )

    rooms_meshed = butil.get_collection("placeholders:room_meshes")
    rooms_split = room_dec.split_rooms(list(rooms_meshed.objects))

    p.run_stage(
        "room_walls", room_dec.room_walls, rooms_split["wall"].objects, use_chance=False
    )
    p.run_stage(
        "room_pillars",
        room_dec.room_pillars,
        state,
        rooms_split["wall"].objects,
        use_chance=False,
    )
    p.run_stage(
        "room_floors",
        room_dec.room_floors,
        rooms_split["floor"].objects,
        use_chance=False,
    )
    p.run_stage(
        "room_ceilings",
        room_dec.room_ceilings,
        rooms_split["ceiling"].objects,
        use_chance=False,
    )

    # state.print()
    state.to_json(output_folder / "solve_state.json")

    cam = cam_util.get_camera(0, 0)

    def turn_off_lights():
        for o in bpy.data.objects:
            if o.type == "LIGHT" and not o.data.cycles.is_portal:
                print(f"Deleting {o.name}")
                butil.delete(o)

    p.run_stage("lights_off", turn_off_lights)

    def invisible_room_ceilings():
        rooms_split["exterior"].hide_viewport = True
        rooms_split["exterior"].hide_render = True
        invisible_to_camera.apply(list(rooms_split["ceiling"].objects))
        invisible_to_camera.apply(
            [o for o in bpy.data.objects if "CeilingLight" in o.name]
        )

    p.run_stage("invisible_room_ceilings", invisible_room_ceilings, use_chance=False)

    p.run_stage(
        "hide_other_rooms",
        hide_other_rooms,
        state,
        rooms_split,
        keep_rooms=[r.name for r in solved_rooms],
        use_chance=False,
    )

    height = p.run_stage(
        "nature_backdrop",
        create_outdoor_backdrop,
        terrain,
        house_bbox=house_bbox,
        cam=cam,
        p=p,
        params=overrides,
        use_chance=False,
        prereq="terrain",
        default=0,
    )

    if overrides.get("topview", False):
        rooms_split["exterior"].hide_viewport = True
        rooms_split["ceiling"].hide_viewport = True
        rooms_split["exterior"].hide_render = True
        rooms_split["ceiling"].hide_render = True
        for group in ["wall", "floor"]:
            for wall in rooms_split[group].objects:
                for mat in wall.data.materials:
                    for n in mat.node_tree.nodes:
                        if n.type == "BSDF_PRINCIPLED":
                            n.inputs["Alpha"].default_value = overrides.get(
                                "alpha_walls", 1.0
                            )
        bbox = np.concatenate(
            [
                read_co(r) + np.array(r.location)[np.newaxis, :]
                for r in rooms_meshed.objects
            ]
        )
        camera = camera_rigs[0].children[0]
        camera_rigs[0].location = 0, 0, 0
        camera_rigs[0].rotation_euler = 0, 0, 0
        bpy.contexScene.camera = camera
        rot_x = deg2rad(overrides.get("topview_rot_x", 0))
        rot_z = deg2rad(overrides.get("topview_rot_z", 0))
        camera.rotation_euler = rot_x, 0, rot_z
        mean = np.mean(bbox, 0)
        for cam_dist in np.exp(np.linspace(1.0, 5.0, 500)):
            camera.location = (
                mean[0] + cam_dist * np.sin(rot_x) * np.sin(rot_z),
                mean[1] - cam_dist * np.sin(rot_x) * np.cos(rot_z),
                mean[2] - WALL_HEIGHT / 2 + cam_dist * np.cos(rot_x),
            )
            bpy.context.view_layer.update()
            inview = points_inview(bbox, camera)
            if inview.all():
                for area in bpy.contexScreen.areas:
                    if area.type == "VIEW_3D":
                        area.spaces.active.region_3d.view_perspective = "CAMERA"
                        break
                break

    return height
