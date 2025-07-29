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
import sys

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
    init_graph,
    light,
    record,
    room_structure,
    update_graph,
)
from infinigen_examples.util import constraint_util as cu
from infinigen_examples.util.generate_indoors_util import (
    restrict_solving,
)
from infinigen_examples.util.visible import (
    invisible_wall,
)
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
import json
import os
import socket
import sys
import threading
import time

import bpy
import gin
import numpy as np

from infinigen.core import execute_tasks, init
from infinigen.core.constraints import checks
from infinigen.core.constraints.constraint_language.util import delete_obj_with_children
from infinigen.core.constraints.example_solver import (
    populate,
)
from infinigen.core.constraints.example_solver.geometry.validity import (
    all_relations_valid,
)
from infinigen.core.constraints.example_solver.room import constants
from infinigen.core.util import pipeline
from infinigen_examples.indoor_constraint_examples import home_constraints
from infinigen_examples.steps import (
    basic_scene,
    camera,
    complete_structure,
    evaluate,
    init_graph,
    light,
    record,
    room_structure,
    solve_objects,
    update_graph,
)
from infinigen_examples.util import constraint_util as cu
from infinigen_examples.util.generate_indoors_util import (
    restrict_solving,
)
from infinigen_examples.util.visible import (
    invisible_others,
    invisible_wall,
    visible_layers,
    visible_others,
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
    if iter == 0 and action != "add_relation":
        p = pipeline.RandomStageExecutor(scene_seed, output_folder, overrides)
        p, terrain = basic_scene.basic_scene(
            scene_seed, output_folder, overrides, logger, p
        )
        os.environ["ROOM_INFO"] = f"{save_dir}/roominfo.json"
        state, solver, override = room_structure.build_room_structure(
            p, overrides, stages, logger, output_folder, scene_seed, consgraph
        )

        light.turn_off(p)

        camera_rigs, solved_rooms, house_bbox, solved_bbox = camera.animate_camera(
            state, stages, limits, solver, p
        )
        view_all()
        state.__post_init__()
        if action == "init_physcene":
            state, solver = init_graph.init_physcene(stages, limits, solver, state, p)

        elif action == "init_metascene":
            state, solver = init_graph.init_metascene(stages, limits, solver, state, p)
        elif action == "init_gpt":
            solver.load_gpt_results()
            state, solver = init_graph.init_gpt(stages, limits, solver, state, p)
        else:
            raise ValueError(f"Action is wrong: {action}")
    else:
        if inplace:
            load_iter = iter
            os.system(
                f"cp {save_dir}/record_scene/render_{iter}_marked.jpg {save_dir}/record_scene/render_{iter}_marked_inplaced.jpg"
            )
            os.system(
                f"cp {save_dir}/record_scene/render_{iter}.jpg {save_dir}/record_scene/render_{iter}_inplaced.jpg"
            )
            os.system(
                f"cp {save_dir}/record_files/metric_{iter}.json {save_dir}/record_files/metric_{iter}_inplaced.json"
            )
            os.system(
                f"cp {save_dir}/record_files/scene_{iter}.blend {save_dir}/record_files/scene_{iter}_inplaced.blend"
            )
            os.system(
                f"cp {save_dir}/record_files/env_{iter}.pkl {save_dir}/record_files/env_{iter}_inplaced.pkl"
            )
            os.system(
                f"cp {save_dir}/record_files/house_bbox_{iter}.pkl {save_dir}/record_files/house_bbox_{iter}_inplaced.pkl"
            )
            os.system(
                f"cp {save_dir}/record_files/p_{iter}.pkl {save_dir}/record_files/p_{iter}_inplaced.pkl"
            )
            os.system(
                f"cp {save_dir}/record_files/solved_bbox_{iter}.pkl {save_dir}/record_files/solved_bbox_{iter}_inplaced.pkl"
            )
            os.system(
                f"cp {save_dir}/record_files/solver_{iter}.pkl {save_dir}/record_files/solver_{iter}_inplaced.pkl"
            )
            os.system(
                f"cp {save_dir}/record_files/state_{iter}.pkl {save_dir}/record_files/state_{iter}_inplaced.pkl"
            )
            os.system(
                f"cp {save_dir}/record_files/terrain_{iter}.pkl {save_dir}/record_files/terrain_{iter}_inplaced.pkl"
            )
            os.system(
                f"cp {save_dir}/pipeline/metric_{iter}.json {save_dir}/pipeline/metric_{iter}_inplaced.json"
            )
        else:
            load_iter = iter - 1
        p = pipeline.RandomStageExecutor(scene_seed, output_folder, overrides)
        state, solver, terrain, house_bbox, solved_bbox, _ = record.load_scene(
            load_iter
        )

        view_all()
        # save_path = "debug1.blend"
        # bpy.ops.wm.save_as_mainfile(filepath=save_path)
        # from infinigen.core.constraints.evaluator.node_impl.trimesh_geometry import accessibility_cost
        # import pdb
        # pdb.set_trace()
        # lst = [ 'ObjaverseCategoryFactory(8977389).spawn_asset(5348940)', 'ObjaverseCategoryFactory(9726544).spawn_asset(155259)', 'SingleCabinetFactory(4185867).spawn_asset(339423)', 'LargePlantContainerFactory(3315598).spawn_asset(4782290)', 'SimpleBookcaseFactory(7072664).spawn_asset(3590337)', 'ObjaverseCategoryFactory(7452565).spawn_asset(232670)', 'SidetableDeskFactory(7921819).spawn_asset(7806247)', 'BottleFactory(3521636).spawn_asset(4028293)', 'BookStackFactory(5842257).spawn_asset(8922245)', 'BookStackFactory(8331176).spawn_asset(6185393)', 'DeskLampFactory(2449225).spawn_asset(5484490)', 'DeskLampFactory(5484490).spawn_asset(3521636)']
        # for  i in range(len(lst)):
        #     cost = accessibility_cost(state.trimesh_scene, lst[i],lst[i+1], visualize=False, fast=True)
        #     print(cost)
        camera_rigs = [bpy.data.objects.get("CameraRigs/0")]
        if action == "add_relation":
            state, solver = update_graph.add_new_relation(solver, state, p)
            # case "solve_large":
            #     state, solver = solve_objects.solve_large_object(
            #         stages, limits, solver, state, p, consgraph, overrides
            #     )
            # case "solve_medium":
            #     state, solver = solve_objects.solve_medium_object(
            #         stages, limits, solver, state, p, consgraph, overrides
            #     )
            # case "solve_large_and_medium":
            #     state, solver = solve_objects.solve_large_and_medium_object(
            #         stages, limits, solver, state, p, consgraph, overrides
            #     )
        # elif action=="solve_small":
        #     state, solver = solve_objects.solve_small_object(
        #         stages, limits, solver, state, p, consgraph, overrides
        #     )
        elif action == "remove_object":
            state = update_graph.remove_object(solver, state, p)
        elif action == "add_gpt":
            state, solver = update_graph.add_gpt(stages, limits, solver, state, p)
        elif action == "add_acdc":
            state, solver = update_graph.add_acdc(solver, state, p, description)
        elif action == "add_rule":
            state, solver = update_graph.add_rule(stages, limits, solver, state, p)
        elif action == "add_crowd":
            state, solver = update_graph.add_obj_crowd(solver, state, p)
        elif action == "export_supporter":
            record.export_supporter(
                state,
                obj_name=description,
                export_path=f"{save_dir}/record_files/obj.blend",
            )
            record_success()
            sys.exit()
        elif action == "update":
            state, solver = update_graph.update(solver, state, p)
        elif action == "update_size":
            state, solver = update_graph.update_size(solver, state, p)
        # case "modify":
        #     state, solver = update_graph.modify(stages, limits, solver, p)
        elif action == "finalize_scene":
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
            invisible_wall()
        else:
            raise ValueError(f"Action is wrong: {action}")

    if "nophy" not in save_dir:
        if action not in [
            "init_physcene",
            "init_metascene",
            "finalize_scene",
            "add_acdc",
        ]:
            p.run_stage(
                "populate_assets",
                populate.populate_state_placeholders_mid,
                state,
                use_chance=False,
            )
            if action == "add_relation":
                state, solver = solve_objects.solve_large_object(
                    stages, limits, solver, state, p, consgraph, overrides
                )
            else:
                stop = False
                while not stop:
                    state, solver = solve_objects.solve_large_object(
                        stages, limits, solver, state, p, consgraph, overrides
                    )
                    for k, objinfo in state.objs.items():
                        if hasattr(objinfo, "populate_obj"):
                            asset_obj = bpy.data.objects.get(objinfo.populate_obj)
                            place_obj = objinfo.obj
                            if not np.allclose(asset_obj.location, place_obj.location):
                                a = 1
                            asset_obj.rotation_mode = "XYZ"
                            place_obj.rotation_mode = "XYZ"
                            if not np.allclose(
                                asset_obj.rotation_euler, place_obj.rotation_euler
                            ):
                                a = 1
                    p.run_stage(
                        "populate_assets",
                        populate.populate_state_placeholders_mid,
                        state,
                        update_trimesh=False,
                        use_chance=False,
                    )

                    solver.del_no_relation_objects()
                    # save_path = "debug3.blend"
                    # bpy.ops.twm.save_as_mainfile(filepath=save_path)
                    stop = evaluate.del_top_collide_obj(state, iter)

                    solver.del_no_relation_objects()

                    if not bpy.app.background:
                        invisible_others()
                        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
                        visible_others()

                for name in list(state.objs.keys())[::-1]:
                    if name in state.objs.keys():
                        if name != "newroom_0-0":
                            if not all_relations_valid(
                                state, name, use_initial=True, fix_pos=True
                            ):
                                if check_support(state, name, ratio=0.6):
                                    # continue if children object is supported > 60%
                                    continue
                                print("all_relations_valid not valid ", name)
                                objname = state.objs[name].obj.name
                                delete_obj_with_children(
                                    state.trimesh_scene,
                                    objname,
                                    delete_blender=True,
                                    delete_asset=True,
                                )
                                state.objs.pop(name)
                    if not bpy.app.background:
                        invisible_others(hide_placeholder=True)
                        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
                        visible_others()
                solver.del_no_relation_objects()

    # state,solver = solve_objects.solve_medium_object(stages,limits,solver,state,p,consgraph,overrides)
    # state,solver = solve_objects.solve_small_object(stages,limits,solver,state,p,consgraph,overrides)
    record.record_scene(
        state, solver, terrain, house_bbox, solved_bbox, camera_rigs, iter, p
    )

    # evaluate.eval_metric(state, iter, remove_bad=True)

    record_success()

    # save_path = "debug.blend"
    # bpy.ops.wm.save_as_mainfile(filepath=save_path)
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
