import numpy as np

from infinigen.core import placement, tagging
from infinigen.core import tags as t
from infinigen.core.constraints import reasoning as r
from infinigen.core.constraints.example_solver import (
    greedy,
)
from infinigen.core.placement import camera as cam_util
from infinigen.core.util import blender as butil
from infinigen_examples.util import constraint_util as cu


def animate_camera(state, stages, limits, solver, p):
    # region animate cameras
    solved_rooms = [
        state.objs[assignment[cu.variable_room]].obj
        for assignment in greedy.iterate_assignments(
            stages["on_floor"], state, [cu.variable_room], limits
        )
    ]
    solved_bound_points = np.concatenate([butil.bounds(r) for r in solved_rooms])
    solved_bbox = (
        np.min(solved_bound_points, axis=0),
        np.max(solved_bound_points, axis=0),
    )

    house_bbox = np.concatenate(
        [
            butil.bounds(obj)
            for obj in solver.get_bpy_objects(r.Domain({t.Semantics.Room}))
        ]
    )
    house_bbox = (np.min(house_bbox, axis=0), np.max(house_bbox, axis=0))

    camera_rigs = placement.camera.spawn_camera_rigs()

    def pose_cameras():
        nonroom_objs = [
            o.obj for o in state.objs.values() if t.Semantics.Room not in o.tags
        ]
        scene_objs = solved_rooms + nonroom_objs

        scene_preprocessed = placement.camera.camera_selection_preprocessing(
            terrain=None, scene_objs=scene_objs
        )

        solved_floor_surface = butil.join_objects(
            [
                tagging.extract_tagged_faces(o, {t.Subpart.SupportSurface})
                for o in solved_rooms
            ]
        )

        placement.camera.configure_cameras(
            camera_rigs,
            scene_preprocessed=scene_preprocessed,
            init_surfaces=solved_floor_surface,
        )

        return scene_preprocessed

    scene_preprocessed = p.run_stage("pose_cameras", pose_cameras, use_chance=False)

    # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    def animate_cameras():
        cam_util.animate_cameras(camera_rigs, solved_bbox, scene_preprocessed, pois=[])

    p.run_stage(
        "animate_cameras", animate_cameras, use_chance=False, prereq="pose_cameras"
    )
    cam = cam_util.get_camera(0, 0)
    # endregion

    return camera_rigs, solved_rooms, house_bbox, solved_bbox
