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

import bpy
import gin

from infinigen.core import execute_tasks, init
from infinigen.core.constraints import checks
from infinigen.core.constraints.example_solver.room import constants
from infinigen.core.util import pipeline
from infinigen_examples.indoor_constraint_examples import home_constraints
from infinigen_examples.steps import (
    basic_scene,
    camera,
    complete_structure,
    light,
    room_structure,
)
from infinigen_examples.util import constraint_util as cu
from infinigen_examples.util.generate_indoors_util import (
    restrict_solving,
)

logger = logging.getLogger(__name__)

all_vars = [cu.variable_room, cu.variable_obj]


def view_all():
    if not bpy.app.background:
        for area in bpy.context.screen.areas:
            if area.type == "VIEW_3D":
                for region in area.regions:
                    if region.type == "WINDOW":
                        override = {"area": area, "region": region}
                        bpy.ops.view3d.view_all(override, center=True)


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
    height = 1

    consgraph = home_constraints()
    stages = basic_scene.default_greedy_stages()
    checks.check_all(consgraph, stages, all_vars)

    stages, consgraph, limits = restrict_solving(stages, consgraph)

    # p = pipeline.RandomStageExecutor(scene_seed, output_folder, overrides)
    os.environ["JSON_RESULTS"] = json_name
    save_dir = os.getenv("save_dir")

    # if iter == 0 and action != "add_relation":
    if True:
        p = pipeline.RandomStageExecutor(scene_seed, output_folder, overrides)
        p, terrain = basic_scene.basic_scene(
            scene_seed, output_folder, overrides, logger, p
        )
        os.environ["ROOM_INFO"] = "roominfo.json"
        state, solver, override = room_structure.build_room_structure(
            p, overrides, stages, logger, output_folder, scene_seed, consgraph
        )

        light.turn_off(p)

        camera_rigs, solved_rooms, house_bbox, solved_bbox = camera.animate_camera(
            state, stages, limits, solver, p
        )
        view_all()
        state.__post_init__()

        solved_rooms = [bpy.data.objects["newroom_0-0"]]

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
        # invisible_wall()

    save_dir = os.getenv("save_dir")
    save_path = f"{save_dir}/record_files/scene_{iter}.blend"
    bpy.ops.file.make_paths_absolute()
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=save_path, check_existing=False)
    # record.record_scene(
    #     state, solver, terrain, house_bbox, solved_bbox, camera_rigs, iter, p
    # )
    return {
        "height_offset": height,
        "whole_bbox": house_bbox,
    }


def check_support(state, child_name, ratio=0.6):
    import trimesh

    from infinigen.core.constraints.constraint_language import util as iu
    from infinigen_examples.steps.tools import export_relation

    parent_relations = [
        [rel.target_name, export_relation(rel.relation)]
        for rel in state.objs[child_name].relations
    ]
    parent_names = []
    for rel in parent_relations:
        if rel[0] == "newroom_0-0":
            return False
        if rel[1] not in ["on", "ontop"]:
            return False
        parent_names.append(rel[0])
    if len(parent_names) == 0:
        return False
    if len(parent_names) > 1:
        return False

    scene = state.trimesh_scene
    sa = state.objs[child_name]
    sb = state.objs[parent_names[0]]

    a_trimesh = iu.meshes_from_names(scene, sa.obj.name)[0]
    b_trimesh = iu.meshes_from_names(scene, sb.obj.name)[0]

    normal_b = [0, 0, 1]
    origin_b = [0, 0, 0]
    projected_a = trimesh.path.polygons.projected(a_trimesh, normal_b, origin_b)
    projected_b = trimesh.path.polygons.projected(b_trimesh, normal_b, origin_b)

    intersection = projected_a.intersection(projected_b)
    if intersection.area / projected_a.area > ratio:
        return True
    else:
        return False


def has_relation_with_obj(state, child_name):
    from infinigen_examples.steps.tools import export_relation

    parent_relations = [
        [rel.target_name, export_relation(rel.relation)]
        for rel in state.objs[child_name].relations
    ]
    parent_names = []
    for rel in parent_relations:
        if rel[0] == "newroom_0-0":
            continue
        if rel[1] not in ["on", "ontop"]:
            continue
        parent_names.append(rel[0])
    if len(parent_names) == 0:
        return False
    else:
        return True


def record_success():
    save_dir = os.getenv("save_dir")
    with open(f"{save_dir}/args.json", "r") as f:
        j = json.load(f)

    with open(f"{save_dir}/args.json", "w") as f:
        j["success"] = True
        json.dump(j, f, indent=4)
    return


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
    parser.add_argument("--save_dir", type=str, default="debug/")
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

 
    save_dir = args.save_dir
    os.environ["save_dir"] = save_dir

    with open(f"{save_dir}/args.json", "r") as f:
        j = json.load(f)
        args.iter = j["iter"]
        args.action = j["action"]
        args.description = j["description"]
        args.inplace = j["inplace"]
        args.json_name = j["json_name"]

    if not os.path.exists(f"{save_dir}/args"):
        os.system(f"mkdir {save_dir}/args")
        os.system(f"mkdir {save_dir}/record_files")
        os.system(f"mkdir {save_dir}/record_scene")
    if args.inplace:
        os.system(
            f"cp {save_dir}/args/args_{args.iter}.json {save_dir}/args/args_{args.iter}_inplaced.json"
        )
    os.system(f"cp {save_dir}/args.json {save_dir}/args/args_{args.iter}.json")

    main(args)
