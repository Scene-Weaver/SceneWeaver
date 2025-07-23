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
from infinigen.core.constraints import checks
from infinigen.core.constraints.example_solver.room import constants
from infinigen_examples.indoor_constraint_examples import home_constraints
from infinigen_examples.steps import (
    basic_scene,
    evaluate,
    record,
)
from infinigen_examples.util import constraint_util as cu
from infinigen_examples.util.generate_indoors_util import (
    restrict_solving,
)

logger = logging.getLogger(__name__)

all_vars = [cu.variable_room, cu.variable_obj]


@gin.configurable
def compose_indoors(
    output_folder: Path,
    scene_seed: int,
    iter,
    action,
    json_name,
    description,
    inplace,
    **overrides,
):
    for iter in range(4, 16):
        height = 1

        consgraph = home_constraints()
        stages = basic_scene.default_greedy_stages()
        checks.check_all(consgraph, stages, all_vars)

        stages, consgraph, limits = restrict_solving(stages, consgraph)
        os.environ["JSON_RESULTS"] = json_name

        load_iter = iter
        state, solver, terrain, house_bbox, solved_bbox, p = record.load_scene(
            load_iter
        )
        evaluate.eval_metric(state, iter)

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
        iter=args.iter,
        action=args.action,
        json_name=args.json_name,
        description=args.description,
        inplace=args.inplace,
        populate_scene_func=None,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        task=args.task,
        task_uniqname=args.task_uniqname,
        scene_seed=scene_seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--action", type=str, default="init_physcene")
    parser.add_argument("--json_name", type=str, default="")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--inplace", type=str, default="")
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

    # invisible_others()
    # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    # visible_others()

    args = init.parse_args_blender(parser)
    logging.getLogger("infinigen").setLevel(logging.INFO)
    logging.getLogger("infinigen.core.nodes.node_wrangler").setLevel(logging.CRITICAL)

    if args.debug is not None:
        for name in logging.root.manager.loggerDict:
            if not name.startswith("infinigen"):
                continue
            if len(args.debug) == 0 or any(name.endswith(x) for x in args.debug):
                logging.getLogger(name).setLevel(logging.DEBUG)

    import json

    with open("args.json", "r") as f:
        j = json.load(f)
        args.iter = j["iter"]
        args.action = j["action"]
        args.description = j["description"]
        args.inplace = j["inplace"]
        args.json_name = j["json_name"]

    os.system(f"cp args.json args_{args.iter}.json")

    main(args)
