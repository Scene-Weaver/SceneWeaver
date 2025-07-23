# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

import argparse
import logging
from pathlib import Path

# ruff: noqa: E402
# NOTE: logging config has to be before imports that use logging
logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
import os

import gin

from infinigen.core import execute_tasks, init
from infinigen.core.constraints.example_solver.room import constants
from infinigen_examples.util import constraint_util as cu

# from . import generate_nature  # noqa F401 # needed for nature gin configs to load

logger = logging.getLogger(__name__)

all_vars = [cu.variable_room, cu.variable_obj]


@gin.configurable
def compose_indoors(output_folder: Path, scene_seed: int, **overrides):
    from .steps import (
        basic_scene,
        camera,
        complete_structure,
        init_graph,
        light,
        populate_placeholder,
        record,
        room_structure,
        solve_objects,
        update_graph,
    )

    p, stages, consgraph, limits, terrain = basic_scene.basic_scene(
        scene_seed, output_folder, overrides, logger
    )

    os.environ["GPT_RESULTS"] = (
        "GPT/results_classroom_gpt_turbo.json"
    )
    state, solver, override = room_structure.build_room_structure(
        p, overrides, stages, logger, output_folder, scene_seed, consgraph
    )

    state, solver = init_graph.init_physcene(stages, limits, solver, state, p)
    # state,solver = init_graph.init_gpt(stages,limits,solver,state,p)
    # state,solver = init_graph.init_metascene(stages,limits,solver,state,p)

    state, solver = solve_objects.solve_large_object(
        stages, limits, solver, state, p, consgraph, overrides
    )
    camera_rigs, solved_rooms, house_bbox, solved_bbox = camera.animate_camera(
        state, stages, limits, solver, p
    )

    # populate_placeholder.populate_intermediate_pholders(p,solver)
    light.turn_off(p)

    state, solver = update_graph.add_gpt(stages, limits, solver, p)
    record.record_scene(state, solver, solved_bbox, camera_rigs, p)

    state, solver = solve_objects.solve_medium_object(
        stages, limits, solver, state, p, consgraph, overrides
    )
    state, solver = update_graph.modify(stages, limits, solver, p)
    state, solver = update_graph.update(stages, limits, solver, p)

    state, solver = solve_objects.solve_large_and_medium_object(
        stages, limits, solver, state, p, consgraph, overrides
    )

    populate_placeholder.populate_intermediate_pholders(p, solver)

    record.export_supporter(state)

    state, solver = update_graph.add_acdc(solver, p)
    state, solver = solve_objects.solve_small_object(
        stages, limits, solver, state, p, consgraph, overrides
    )
    populate_placeholder.populate_intermediate_pholders(p, state)

    height = complete_structure.finalize_scene(
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
    )

    return {
        "height_offset": height,
        "whole_bbox": house_bbox,
    }


def main(args):
    scene_seed = init.apply_scene_seed(args.seed)
    init.apply_gin_configs(
        configs=["base_indoors.gin"] + args.configs,
        overrides=args.overrides,
        config_folders=[
            "infinigen_examples/configs_indoor",
            "infinigen_examples/configs_nature",
        ],
    )
    constants.initialize_constants()

    execute_tasks.main(
        compose_scene_func=compose_indoors,
        populate_scene_func=None,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        task=args.task,
        task_uniqname=args.task_uniqname,
        scene_seed=scene_seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=Path)
    parser.add_argument("--input_folder", type=Path, default=None)
    parser.add_argument(
        "-s", "--seed", default=None, help="The seed used to generate the scene"
    )
    parser.add_argument(
        "-t",
        "--task",
        nargs="+",
        default=["coarse"],
        choices=[
            "coarse",
            "populate",
            "fine_terrain",
            "ground_truth",
            "render",
            "mesh_save",
            "export",
        ],
    )
    parser.add_argument(
        "-g",
        "--configs",
        nargs="+",
        default=["base"],
        help="Set of config files for gin (separated by spaces) "
        "e.g. --gin_config file1 file2 (exclude .gin from path)",
    )
    parser.add_argument(
        "-p",
        "--overrides",
        nargs="+",
        default=[],
        help="Parameter settings that override config defaults "
        "e.g. --gin_param module_1.a=2 module_2.b=3",
    )
    parser.add_argument("--task_uniqname", type=str, default=None)
    parser.add_argument("-d", "--debug", type=str, nargs="*", default=None)

    args = init.parse_args_blender(parser)
    logging.getLogger("infinigen").setLevel(logging.INFO)
    logging.getLogger("infinigen.core.nodes.node_wrangler").setLevel(logging.CRITICAL)

    if args.debug is not None:
        for name in logging.root.manager.loggerDict:
            if not name.startswith("infinigen"):
                continue
            if len(args.debug) == 0 or any(name.endswith(x) for x in args.debug):
                logging.getLogger(name).setLevel(logging.DEBUG)

    # import match
    # match.debug()
    main(args)
