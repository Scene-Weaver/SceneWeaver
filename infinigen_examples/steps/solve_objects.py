from infinigen.core.constraints.example_solver import (
    greedy,
)

from .basic_scene import all_vars


def solve_large_object(stages, limits, solver, state, p, consgraph, overrides):
    # region solve large
    def solve_large():
        assignments = greedy.iterate_assignments(
            stages["on_floor"], state, all_vars, limits, nonempty=True
        )
        for i, vars in enumerate(assignments):
            solver.solve_objects(
                consgraph,
                stages["on_floor"],
                var_assignments=vars,
                n_steps=overrides["solve_steps_large"],
                desc=f"on_floor_{i}",
                abort_unsatisfied=overrides.get("abort_unsatisfied_large", False),
                expand_collision=True,
            )
            break
            # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        return solver.state

    state = p.run_stage("solve_large", solve_large, use_chance=False, default=state)
    # endregion

    return state, solver


def solve_medium_object(stages, limits, solver, state, p, consgraph, overrides):
    # region solve medium
    def solve_medium():
        n_steps = overrides["solve_steps_medium"]
        for i, vars in enumerate(
            greedy.iterate_assignments(stages["on_wall"], state, all_vars, limits)
        ):
            solver.solve_objects(
                consgraph, stages["on_wall"], vars, n_steps, desc=f"on_wall_{i}"
            )
        for i, vars in enumerate(
            greedy.iterate_assignments(stages["on_ceiling"], state, all_vars, limits)
        ):
            solver.solve_objects(
                consgraph, stages["on_ceiling"], vars, n_steps, desc=f"on_ceiling_{i}"
            )
        for i, vars in enumerate(
            greedy.iterate_assignments(stages["side_obj"], state, all_vars, limits)
        ):
            solver.solve_objects(
                consgraph,
                stages["side_obj"],
                vars,
                n_steps,
                desc=f"side_obj_{i}",
                expand_collision=True,
                use_initial=True,
            )

        return solver.state

    state = p.run_stage("solve_medium", solve_medium, use_chance=False, default=state)
    # endregion
    return state, solver


def solve_large_and_medium_object(
    stages, limits, solver, state, p, consgraph, overrides
):
    # region solve_large_and_medium
    def solve_large_and_medium():
        for i in range(3):
            assignments = greedy.iterate_assignments(
                stages["on_floor"], state, all_vars, limits, nonempty=True
            )
            for j, vars in enumerate(assignments):
                solver.solve_objects(
                    consgraph,
                    stages["on_floor"],
                    var_assignments=vars,
                    n_steps=overrides["solve_steps_large"] // 5,
                    desc=f"on_floor_{j}",
                    abort_unsatisfied=overrides.get("abort_unsatisfied_large", False),
                    expand_collision=True,
                )
                # invisible_others()
                # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                # visible_others()

            for k, vars in enumerate(
                greedy.iterate_assignments(stages["side_obj"], state, all_vars, limits)
            ):
                n_steps = overrides["solve_steps_medium"] // 5
                solver.solve_objects(
                    consgraph,
                    stages["side_obj"],
                    vars,
                    n_steps,
                    desc=f"side_obj_{k}",
                    expand_collision=True,
                    use_initial=True,
                )

        return solver.state

    state = p.run_stage(
        "solve_large_and_medium",
        solve_large_and_medium,
        use_chance=False,
        default=state,
    )

    return state, solver


def solve_small_object(stages, limits, solver, state, p, consgraph, overrides):
    def solve_small():
        n_steps = overrides["solve_steps_small"]

        for i, vars in enumerate(
            greedy.iterate_assignments(stages["obj_ontop_obj"], state, all_vars, limits)
        ):
            solver.solve_objects(
                consgraph,
                stages["obj_ontop_obj"],
                vars,
                n_steps,
                expand_collision=True,
                desc=f"obj_ontop_obj_{i}",
            )
        for i, vars in enumerate(
            greedy.iterate_assignments(
                stages["obj_on_support"], state, all_vars, limits
            )
        ):
            solver.solve_objects(
                consgraph,
                stages["obj_on_support"],
                vars,
                n_steps,
                expand_collision=True,
                desc=f"obj_on_support_{i}",
            )

        return solver.state

    state = p.run_stage("solve_small", solve_small, use_chance=False, default=state)

    return state, solver
