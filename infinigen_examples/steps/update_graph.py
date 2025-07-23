from infinigen.core.constraints.example_solver import (
    greedy,
)

from .basic_scene import all_vars


def add_gpt(stages, limits, solver, state, p):
    # region update by GPT
    # implement by GPT
    def add_graph(this_stage, iter):
        assignments = greedy.iterate_assignments(
            stages["on_floor"], state, all_vars, limits, nonempty=True
        )
        for i, vars in enumerate(assignments):
            solver.add_graph_gpt(
                # stages["on_floor"],
                iter=iter,
                var_assignments=vars,
                stage=this_stage,
            )
            # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        return solver.state

    state = p.run_stage(
        "add_graph",
        add_graph,
        this_stage="large",
        iter=1,
        use_chance=False,
        default=state,
    )
    # endregion
    return state, solver


# def modify(solver, state, p):
#     def modify_graph():
#         solver.modify_graph()
#         return solver.state

#     state = p.run_stage("modify_graph", modify_graph, use_chance=False, default=state)
#     return state


def remove_object(solver, state, p):
    def remove_obj():
        solver.remove_object()
        return solver.state

    state = p.run_stage("remove_object", remove_obj, use_chance=False, default=state)
    return state


def add_new_relation(solver, state, p):
    def add_relation():
        solver.add_relation()
        return solver.state

    state = p.run_stage("add_relation", add_relation, use_chance=False, default=state)
    return state, solver


def update(solver, state, p):
    def update_graph():
        solver.update_graph()
        return solver.state

    state = p.run_stage("update_graph", update_graph, use_chance=False, default=state)
    return state, solver


def update_size(solver, state, p):
    def update_graph_size():
        solver.update_graph_size()
        return solver.state

    state = p.run_stage(
        "update_graph", update_graph_size, use_chance=False, default=state
    )
    return state, solver


def add_acdc(solver, state, p, description):
    # region load acdc
    def load_acdc():
        solver.load_acdc(parent_obj_name=description)
        return solver.state

    state = p.run_stage("load_acdc", load_acdc, use_chance=False, default=state)
    # endregion
    return state, solver


def add_rule(solver, state, p):
    # region load acdc
    def add_rule():
        solver.add_rule()
        return solver.state

    state = p.run_stage("load_acdc", add_rule, use_chance=False, default=state)
    # endregion
    return state, solver


def add_obj_crowd(solver, state, p):
    # region load acdc
    def obj_crowd():
        solver.add_obj_crowd(
            iter=iter,
            var_assignments=vars,
        )
        return solver.state

    state = p.run_stage("add_obj_crowd", obj_crowd, use_chance=False, default=state)
    # endregion
    return state, solver
