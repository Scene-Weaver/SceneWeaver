import bpy

from infinigen.core import tags as t
from infinigen.core.constraints.example_solver import (
    Solver,
    state_def,
)
from infinigen_examples.util import constraint_util as cu
from infinigen_examples.util.generate_indoors_util import (
    apply_greedy_restriction,
)


def build_room_structure(
    p, overrides, stages, logger, output_folder, scene_seed, consgraph
):
    # region room structure
    if overrides.get("restrict_single_supported_roomtype", False):
        restrict_parent_rooms = t.Semantics.NewRoom

        logger.info(f"Restricting to {restrict_parent_rooms}")
        apply_greedy_restriction(stages, restrict_parent_rooms, cu.variable_room)

    solver = Solver(output_folder=output_folder)

    def solve_rooms():
        return solver.solve_rooms(scene_seed, consgraph, stages["rooms"])

    state: state_def.State = p.run_stage("solve_rooms", solve_rooms, use_chance=False)

    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            with bpy.context.temp_override(area=area):
                area.spaces.active.shading.type = "MATERIAL"
            for region in area.regions:
                if region.type == "WINDOW":
                    override = {
                        "area": area,
                        "region": region,
                        "edit_object": bpy.context.edit_object,
                    }
                    bpy.ops.view3d.view_all(override)

    # endregion
    return state, solver, override
