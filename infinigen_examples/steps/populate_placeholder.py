from infinigen.core import tags as t
from infinigen.core.constraints.example_solver import (
    populate,
)


def populate_intermediate_pholders(p, solver):
    # region populate_intermediate_pholders

    p.run_stage(
        "populate_intermediate_pholders",
        populate.populate_state_placeholders,
        solver.state,
        filter=t.Semantics.AssetPlaceholderForChildren,
        final=False,
        use_chance=False,
    )

    # endregion
    return


def populate_intermediate_pholders(p, state):
    p.run_stage(
        "populate_assets", populate.populate_state_placeholders, state, use_chance=False
    )
    return
