from infinigen.assets import lighting
from infinigen.core import surface
from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import reasoning as r
from infinigen.terrain import Terrain
from infinigen_examples.util import constraint_util as cu

all_vars = [cu.variable_room, cu.variable_obj]


def default_greedy_stages():
    """Returns descriptions of what will be covered by each greedy stage of the solver.

    Any domain containing one or more VariableTags is greedy: it produces many separate domains,
        one for each possible assignment of the unresolved variables.
    """

    on_floor = cl.StableAgainst({}, cu.floortags)
    on_wall = cl.StableAgainst({}, cu.walltags)
    on_ceiling = cl.StableAgainst({}, cu.ceilingtags)
    side = cl.StableAgainst({}, cu.side)

    all_room = r.Domain({t.Semantics.Room, -t.Semantics.Object})
    all_obj = r.Domain({t.Semantics.Object, -t.Semantics.Room})

    all_obj_in_room = all_obj.with_relation(
        cl.AnyRelation(), all_room.with_tags(cu.variable_room)
    )
    primary = all_obj_in_room.with_relation(-cl.AnyRelation(), all_obj)

    greedy_stages = {}

    greedy_stages["rooms"] = all_room

    greedy_stages["on_floor"] = primary.with_relation(on_floor, all_room)
    greedy_stages["on_wall"] = (
        primary.with_relation(-on_floor, all_room)
        .with_relation(-on_ceiling, all_room)
        .with_relation(on_wall, all_room)
    )
    greedy_stages["on_ceiling"] = (
        primary.with_relation(-on_floor, all_room)
        .with_relation(on_ceiling, all_room)
        .with_relation(-on_wall, all_room)
    )

    secondary = all_obj.with_relation(
        cl.AnyRelation(), primary.with_tags(cu.variable_obj)
    )

    greedy_stages["side_obj"] = secondary.with_relation(side, all_obj)
    nonside = secondary.with_relation(-side, all_obj)

    greedy_stages["obj_ontop_obj"] = nonside.with_relation(
        cu.ontop, all_obj
    ).with_relation(-cu.on, all_obj)
    greedy_stages["obj_on_support"] = nonside.with_relation(
        cu.on, all_obj
    ).with_relation(-cu.ontop, all_obj)

    return greedy_stages


def basic_scene(scene_seed, output_folder, overrides, logger, p):
    logger.debug(overrides)

    def add_coarse_terrain():
        terrain = Terrain(
            scene_seed,
            surface.registry,
            task="coarse",
            on_the_fly_asset_folder=output_folder / "assets",
        )
        terrain_mesh = terrain.coarse_terrain()
        # placement.density.set_tag_dict(terrain.tag_dict)
        return terrain, terrain_mesh

    terrain, terrain_mesh = p.run_stage(
        "terrain", add_coarse_terrain, use_chance=False, default=(None, None)
    )

    p.run_stage("sky_lighting", lighting.sky_lighting.add_lighting, use_chance=False)

    return p, terrain
