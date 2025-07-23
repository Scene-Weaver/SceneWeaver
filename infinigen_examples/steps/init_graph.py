from infinigen.core.constraints.example_solver import (
    greedy,
)

from .basic_scene import all_vars


def init_physcene(stages, limits, solver, state, p):
    # region init large physcene
    def init_graph_physcene(this_stage):
        assignments = greedy.iterate_assignments(
            stages["on_floor"], state, all_vars, limits, nonempty=True
        )
        for i, vars in enumerate(assignments):
            solver.init_graph_physcene(
                # solver.init_graph_gpt(
                # stages["on_floor"],
                var_assignments=vars,
                stage=this_stage,
            )
        return solver.state

    state = p.run_stage(
        "init_graph_physcene",
        init_graph_physcene,
        this_stage="large",
        use_chance=False,
        default=state,
    )
    # endregion
    return state, solver


def init_idesign(stages, limits, solver, state, p):
    solver.init_graph_idesign()
    return solver.state, solver


def init_atiss(stages, limits, solver, state, p):
    solver.init_graph_atiss()
    return solver.state, solver


def init_layoutgpt(stages, limits, solver, state, p):
    solver.init_graph_layoutgpt()
    return solver.state, solver


def init_anyhome(stages, limits, solver, state, p):
    solver.init_graph_anyhome()
    return solver.state, solver


def init_holodeck(stages, limits, solver, state, p):
    solver.init_graph_holodeck()
    return solver.state, solver


def init_gpt(stages, limits, solver, state, p):
    # region init large physcene
    def init_graph_gpt():
        solver.init_graph_gpt(
            var_assignments=vars,
        )
        return solver.state

    state = p.run_stage(
        "init_graph_gpt",
        init_graph_gpt,
        use_chance=False,
        default=state,
    )

    return state, solver


def init_metascene(stages, limits, solver, state, p):
    # region init large physcene
    def init_graph_metascene(this_stage):
        assignments = greedy.iterate_assignments(
            stages["on_floor"], state, all_vars, limits, nonempty=True
        )
        for i, vars in enumerate(assignments):
            solver.init_graph_metascene(
                # solver.init_graph_gpt(
                # stages["on_floor"],
                var_assignments=vars,
                stage=this_stage,
            )
        return solver.state

    state = p.run_stage(
        "init_graph_metascene",
        init_graph_metascene,
        this_stage="large",
        use_chance=False,
        default=state,
    )
    # endregion
    return state, solver
