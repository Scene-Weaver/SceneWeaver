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

# Global variables for socket communication
socket_server = None
is_listening = True
action_queue = []
action_lock = threading.Lock()
global_overrides = []  # Store initial overrides from command line
global_configs = ["base"]  # Store initial configs from command line
command_results = {}


def view_all():
    if not bpy.app.background:
        for area in bpy.context.screen.areas:
            if area.type == "VIEW_3D":
                for region in area.regions:
                    if region.type == "WINDOW":
                        override = {"area": area, "region": region}
                        bpy.ops.view3d.view_all(override, center=True)


class SocketServer:
    def __init__(self, host="localhost", port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False

    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.running = True
            logger.info(f"Socket server started on {self.host}:{self.port}")

            while self.running:
                try:
                    self.socket.settimeout(1.0)  # Allow for periodic checks
                    client_socket, address = self.socket.accept()
                    logger.info(f"Connection from {address}")

                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client, args=(client_socket,)
                    )
                    client_thread.daemon = True
                    client_thread.start()

                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Socket error: {e}")
                    break

        except Exception as e:
            logger.error(f"Failed to start socket server: {e}")
        finally:
            if self.socket:
                self.socket.close()

    def handle_client(self, client_socket):
        try:
            while self.running:
                data = client_socket.recv(1024)
                if not data:
                    break

                try:
                    command = json.loads(data.decode("utf-8"))
                    logger.info(f"Received command: {command}")

                    # Handle special commands immediately
                    action = command.get("action", "")
                    if action in ["ping", "status", "stop_server"]:
                        result = process_action_command(command)
                        response = {"status": "completed", "result": result}
                        client_socket.send(json.dumps(response).encode("utf-8"))
                        if action == "stop_server":
                            break
                    else:
                        # Add command to action queue for processing
                        command_id = len(action_queue)  # Simple ID assignment
                        command["command_id"] = command_id

                        with action_lock:
                            action_queue.append(command)

                        # # Send acknowledgment
                        # response = {"status": "queued", "command_id": command_id, "message": "Command queued for processing"}
                        # client_socket.send(json.dumps(response).encode('utf-8'))
                        # 3. 阻塞等待结果
                        while self.running:
                            if command_id in command_results:
                                result = command_results.pop(command_id)
                                response = {
                                    "status": "completed",
                                    "command_id": command_id,
                                    "result": result,
                                }
                                client_socket.send(json.dumps(response).encode("utf-8"))
                                break
                            time.sleep(0.1)

                except json.JSONDecodeError as e:
                    error_response = {
                        "status": "error",
                        "message": f"Invalid JSON: {e}",
                    }
                    client_socket.send(json.dumps(error_response).encode("utf-8"))

        except Exception as e:
            logger.error(f"Client handling error: {e}")
        finally:
            client_socket.close()

    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()


def start_socket_server():
    global socket_server
    socket_server = SocketServer()
    server_thread = threading.Thread(target=socket_server.start)
    server_thread.daemon = True
    server_thread.start()
    return socket_server


def get_next_action():
    """Get the next action from the queue"""
    with action_lock:
        if action_queue:
            return action_queue.pop(0)
    return None


def update_global_overrides(new_overrides):
    """Update the global overrides at runtime"""
    global global_overrides
    global_overrides = new_overrides[:]
    logger.info(f"Updated global overrides: {global_overrides}")


def add_global_override(override):
    """Add a single override to the global overrides"""
    global global_overrides
    if override not in global_overrides:
        global_overrides.append(override)
        logger.info(
            f"Added override '{override}'. Global overrides: {global_overrides}"
        )


def process_action_command(command):
    """Process a single action command"""
    try:
        # Extract parameters from command
        action = command.get("action", "init_physcene")
        iter_num = command.get("iter", 0)
        description = command.get("description", "")
        inplace = command.get("inplace", False)
        json_name = command.get(
            "json_name",
            "~/workspace/PhyScene/3D_front/generate_filterGPN_clean/Bedroom-47007_bedroom.json",
        )
        seed = command.get("seed", 0)
        # output_folder = command.get('save_dir', "debug/")
        # Set save_dir from command if provided
        save_dir = command.get("save_dir", os.getenv("save_dir", "debug/"))
        os.environ["save_dir"] = save_dir

        logger.info(f"Processing action: {action} with iter: {iter_num}")

        # Handle special commands
        if action == "stop_server":
            return stop_server_command()
        elif action == "ping":
            return {"status": "pong", "message": "Server is running"}
        elif action == "status":
            return {
                "status": "running",
                "queue_size": len(action_queue),
                "message": "Socket server is active and listening",
            }

        # Update parameters from save_dir/args.json if it exists
        args_json_path = f"{save_dir}/args.json"

        if os.path.exists(args_json_path):
            if inplace and os.path.exists(f"{save_dir}/args/args_{iter_num}.json"):
                os.system(
                    f"cp {save_dir}/args/args_{iter_num}.json {save_dir}/args/args_{iter_num}_inplaced.json"
                )
            # Save current args
            os.system(f"cp {args_json_path} {save_dir}/args/args_{iter_num}.json")

            try:
                with open(args_json_path, "r") as f:
                    j = json.load(f)
                    # Update with saved values, but prioritize command values
                    if "iter" not in command:
                        iter_num = j.get("iter", iter_num)
                    if "action" not in command:
                        action = j.get("action", action)
                    if "description" not in command:
                        description = j.get("description", description)
                    if "inplace" not in command:
                        inplace = j.get("inplace", inplace)
                    if "json_name" not in command:
                        json_name = j.get("json_name", json_name)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    f"Could not load args.json: {e}. Using command parameters."
                )

        # Create output folder path
        output_path = Path(save_dir)

        # Apply seed if provided
        if seed is not None:
            # scene_seed = init.apply_scene_seed(seed)
            scene_seed = gin.config._CONFIG.get(("OVERALL_SEED", "gin.macro"))["value"]
        else:
            # scene_seed = init.apply_scene_seed(None)
            scene_seed = gin.config._CONFIG.get(("OVERALL_SEED", "gin.macro"))["value"]

        # Initialize gin configs - ensure this works properly
        # Merge global overrides with command overrides
        global global_overrides

        # Start with global overrides, then add command-specific overrides
        overrides = global_overrides[:]

        # Ensure terrain is disabled to avoid landlab dependency issues
        if "compose_indoors.terrain_enabled=False" not in overrides:
            overrides.append("compose_indoors.terrain_enabled=False")

        logger.info(f"Global overrides: {global_overrides}")
        logger.info(f"Merged overrides: {overrides}")

        try:
            # Always re-apply gin configs with the current overrides for socket commands
            # This ensures that the terrain_enabled=False override is properly applied
            global global_configs
            configs_to_load = ["base_indoors.gin"] + global_configs
            logger.info("Applying gin configs for socket command:")
            logger.info(f"  configs: {configs_to_load}")
            logger.info(f"  overrides: {overrides}")

            # Import the gin module to access current config

            # Clear any existing gin config to ensure clean state
            # gin.clear_config()

            # Apply the same configs as the main function would
            init.apply_gin_configs(
                configs=configs_to_load,
                overrides=overrides,
                config_folders=[
                    "infinigen_examples/configs_indoor",
                    "infinigen_examples/configs_nature",
                ],
            )
            constants.initialize_constants()

            # Debug: Check if the lights_off_chance parameter is loaded
            try:
                lights_off_chance = gin.get_configurable_singletons()[
                    "compose_indoors"
                ].__dict__.get("lights_off_chance", "NOT_FOUND")
                logger.info(f"lights_off_chance parameter: {lights_off_chance}")
            except Exception as debug_e:
                logger.warning(f"Could not check lights_off_chance: {debug_e}")

        except Exception as e:
            logger.error(f"Failed to initialize gin configs: {e}")
            import traceback

            traceback.print_exc()
            raise e

        # Set environment variable for JSON results
        os.environ["JSON_RESULTS"] = json_name

        # Call compose_indoors directly with the action parameters
        # Convert overrides list to dict format for function call
        overrides_dict = {}
        for override in overrides:
            if "=" in override:
                key, value = override.split("=", 1)
                # Handle gin-style function parameters
                if key.startswith("compose_indoors."):
                    key = key.replace("compose_indoors.", "")

                # Convert string values to appropriate types
                if value.lower() == "true":
                    overrides_dict[key] = True
                elif value.lower() == "false":
                    overrides_dict[key] = False
                else:
                    try:
                        # Try to convert to number
                        overrides_dict[key] = (
                            float(value) if "." in value else int(value)
                        )
                    except ValueError:
                        overrides_dict[key] = value

        logger.info(
            f"Calling compose_indoors with overrides_dict: {overrides_dict}"
        )

        result = compose_indoors(
            output_folder=output_path,
            scene_seed=scene_seed,
            iter=iter_num,
            action=action,
            json_name=json_name,
            description=description,
            inplace=inplace,
            **overrides_dict,
        )

        return {
            "status": "success",
            "action": action,
            "iter": iter_num,
            "result": result,
        }

    except Exception as e:
        logger.error(f"Error processing action: {e}")
        import traceback

        traceback.print_exc()
        return {"status": "error", "message": str(e), "action": action}


def execute_main_logic(args):
    """Execute the main scene generation logic"""
    global global_overrides, global_configs

    # scene_seed = init.apply_scene_seed(args.seed)
    scene_seed = gin.config._CONFIG.get(("OVERALL_SEED", "gin.macro"))["value"]

    # Use the same configs and overrides as stored globally
    configs_to_load = ["base_indoors.gin"] + global_configs
    logger.info(f"execute_main_logic using configs: {configs_to_load}")
    logger.info(f"execute_main_logic using overrides: {global_overrides}")

    init.apply_gin_configs(
        configs=configs_to_load,
        overrides=global_overrides,
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


def blender_action_handler():
    """Blender timer function to handle actions"""

    action = get_next_action()
    if action:
        logger.info(f"Executing action: {action}")
        result = process_action_command(action)
        logger.info(f"Action completed with result: {result}")
        # 这里写入 command_results
        command_id = action.get("command_id")
        if command_id is not None:
            command_results[command_id] = result

        # Make sure Blender UI updates
        if not bpy.app.background:
            for area in bpy.context.screen.areas:
                area.tag_redraw()

    return 0.1  # Check again in 0.1 seconds


@gin.configurable
def compose_indoors(
    output_folder: Path,
    scene_seed: int,
    iter,
    action,
    json_name,
    description,
    inplace,
    terrain_enabled=False,  # Default to False to avoid landlab issues
    **overrides,
):
    
    visible_layers()
    # bpy.ops.wm.save_as_mainfile(filepath="debug.blend")
    height = 1
    global state, solver, terrain, house_bbox, solved_bbox, camera_rigs, p

    # Add debugging to see if terrain is disabled
    logger.info(f"compose_indoors called with terrain_enabled={terrain_enabled}")
    overrides["terrain_enabled"] = terrain_enabled

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
        camera_rigs = [bpy.data.objects.get("CameraRigs/0")]

        if action == "add_relation":
            state, solver = update_graph.add_new_relation(solver, state, p)
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
            # save_path = "debug.blend"
            # bpy.ops.wm.save_as_mainfile(filepath=save_path)
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
                    save_path = "debug1.blend"
                    bpy.ops.wm.save_as_mainfile(filepath=save_path)
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

    record.record_scene(
        state, solver, terrain, house_bbox, solved_bbox, camera_rigs, iter, p
    )
    evaluate.eval_metric(state, iter, remove_bad=True)

    record_success()

    # save_path = "debug.blend"
    # bpy.ops.wm.save_as_mainfile(filepath=save_path)
    return


# @gin.configurable
# def compose_indoors_debug(
#     output_folder: Path,
#     scene_seed: int,
#     iter,
#     action,
#     json_name,
#     description,
#     inplace,
#     terrain_enabled=False,  # Default to False to avoid landlab issues
#     **overrides,
# ):
#     height = 1

#     # Add debugging to see if terrain is disabled
#     logger.info(f"compose_indoors called with terrain_enabled={terrain_enabled}")
#     overrides["terrain_enabled"] = terrain_enabled

#     consgraph = home_constraints()
#     stages = basic_scene.default_greedy_stages()
#     checks.check_all(consgraph, stages, all_vars)

#     stages, consgraph, limits = restrict_solving(stages, consgraph)

#     # p = pipeline.RandomStageExecutor(scene_seed, output_folder, overrides)
#     os.environ["JSON_RESULTS"] = json_name
#     save_dir = os.getenv("save_dir")
#     if iter == 0 and action != "add_relation":
#         p = pipeline.RandomStageExecutor(scene_seed, output_folder, overrides)
#         p, terrain = basic_scene.basic_scene(
#             scene_seed, output_folder, overrides, logger, p
#         )
#         os.environ["ROOM_INFO"] = f"{save_dir}/roominfo.json"
#         state, solver, override = room_structure.build_room_structure(
#             p, overrides, stages, logger, output_folder, scene_seed, consgraph
#         )

#         light.turn_off(p)

#         camera_rigs, solved_rooms, house_bbox, solved_bbox = camera.animate_camera(
#             state, stages, limits, solver, p
#         )
#         view_all()
#         state.__post_init__()
#         if action == "init_physcene":
#             state, solver = init_graph.init_physcene(stages, limits, solver, state, p)

#         elif action == "init_metascene":
#             state, solver = init_graph.init_metascene(stages, limits, solver, state, p)
#         elif action == "init_gpt":
#             solver.load_gpt_results()
#             state, solver = init_graph.init_gpt(stages, limits, solver, state, p)
#         else:
#             raise ValueError(f"Action is wrong: {action}")
#     else:
#         if inplace:
#             load_iter = iter
#             os.system(
#                 f"cp {save_dir}/record_scene/render_{iter}_marked.jpg {save_dir}/record_scene/render_{iter}_marked_inplaced.jpg"
#             )
#             os.system(
#                 f"cp {save_dir}/record_scene/render_{iter}.jpg {save_dir}/record_scene/render_{iter}_inplaced.jpg"
#             )
#             os.system(
#                 f"cp {save_dir}/record_files/metric_{iter}.json {save_dir}/record_files/metric_{iter}_inplaced.json"
#             )
#             os.system(
#                 f"cp {save_dir}/record_files/scene_{iter}.blend {save_dir}/record_files/scene_{iter}_inplaced.blend"
#             )
#             os.system(
#                 f"cp {save_dir}/record_files/env_{iter}.pkl {save_dir}/record_files/env_{iter}_inplaced.pkl"
#             )
#             os.system(
#                 f"cp {save_dir}/record_files/house_bbox_{iter}.pkl {save_dir}/record_files/house_bbox_{iter}_inplaced.pkl"
#             )
#             os.system(
#                 f"cp {save_dir}/record_files/p_{iter}.pkl {save_dir}/record_files/p_{iter}_inplaced.pkl"
#             )
#             os.system(
#                 f"cp {save_dir}/record_files/solved_bbox_{iter}.pkl {save_dir}/record_files/solved_bbox_{iter}_inplaced.pkl"
#             )
#             os.system(
#                 f"cp {save_dir}/record_files/solver_{iter}.pkl {save_dir}/record_files/solver_{iter}_inplaced.pkl"
#             )
#             os.system(
#                 f"cp {save_dir}/record_files/state_{iter}.pkl {save_dir}/record_files/state_{iter}_inplaced.pkl"
#             )
#             os.system(
#                 f"cp {save_dir}/record_files/terrain_{iter}.pkl {save_dir}/record_files/terrain_{iter}_inplaced.pkl"
#             )
#             os.system(
#                 f"cp {save_dir}/pipeline/metric_{iter}.json {save_dir}/pipeline/metric_{iter}_inplaced.json"
#             )
#         else:
#             load_iter = iter - 1
#         p = pipeline.RandomStageExecutor(scene_seed, output_folder, overrides)
#         state, solver, terrain, house_bbox, solved_bbox, _ = record.load_scene(
#             load_iter
#         )

#         view_all()
#         camera_rigs = [bpy.data.objects.get("CameraRigs/0")]
#         if action == "add_relation":
#             state, solver = update_graph.add_new_relation(solver, state, p)
#             # case "solve_large":
#             #     state, solver = solve_objects.solve_large_object(
#             #         stages, limits, solver, state, p, consgraph, overrides
#             #     )
#             # case "solve_medium":
#             #     state, solver = solve_objects.solve_medium_object(
#             #         stages, limits, solver, state, p, consgraph, overrides
#             #     )
#             # case "solve_large_and_medium":
#             #     state, solver = solve_objects.solve_large_and_medium_object(
#             #         stages, limits, solver, state, p, consgraph, overrides
#             #     )
#         # elif action=="solve_small":
#         #     state, solver = solve_objects.solve_small_object(
#         #         stages, limits, solver, state, p, consgraph, overrides
#         #     )
#         elif action == "remove_object":
#             state = update_graph.remove_object(solver, state, p)
#         elif action == "add_gpt":
#             state, solver = update_graph.add_gpt(stages, limits, solver, state, p)
#         elif action == "add_acdc":
#             state, solver = update_graph.add_acdc(solver, state, p, description)
#         elif action == "add_rule":
#             state, solver = update_graph.add_rule(stages, limits, solver, state, p)
#         elif action == "add_crowd":
#             state, solver = update_graph.add_obj_crowd(solver, state, p)
#         elif action == "export_supporter":
#             record.export_supporter(
#                 state,
#                 obj_name=description,
#                 export_path=f"{save_dir}/record_files/obj.blend",
#             )
#             record_success()
#             return {
#                 "height_offset": height,
#                 "whole_bbox": house_bbox,
#             }
#         elif action == "update":
#             state, solver = update_graph.update(solver, state, p)
#         elif action == "update_size":
#             state, solver = update_graph.update_size(solver, state, p)
#         # case "modify":
#         #     state, solver = update_graph.modify(stages, limits, solver, p)
#         elif action == "finalize_scene":
#             solved_rooms = [bpy.data.objects["newroom_0-0"]]

#             height = complete_structure.finalize_scene(
#                 overrides,
#                 stages,
#                 state,
#                 solver,
#                 output_folder,
#                 p,
#                 terrain,
#                 solved_rooms,
#                 house_bbox,
#                 camera_rigs,
#             )
#             invisible_wall()
#         else:
#             raise ValueError(f"Action is wrong: {action}")

#     if "nophy" not in save_dir:
#         if action not in [
#             "init_physcene",
#             "init_metascene",
#             "finalize_scene",
#             "add_acdc",
#         ]:
#             p.run_stage(
#                 "populate_assets",
#                 populate.populate_state_placeholders_mid,
#                 state,
#                 use_chance=False,
#             )
#             # save_path = "debug.blend"
#             # bpy.ops.wm.save_as_mainfile(filepath=save_path)
#             if action == "add_relation":
#                 state, solver = solve_objects.solve_large_object(
#                     stages, limits, solver, state, p, consgraph, overrides
#                 )
#             else:
#                 stop = False
#                 while not stop:
#                     state, solver = solve_objects.solve_large_object(
#                         stages, limits, solver, state, p, consgraph, overrides
#                     )
#                     save_path = "debug1.blend"
#                     bpy.ops.wm.save_as_mainfile(filepath=save_path)
#                     for k, objinfo in state.objs.items():
#                         if hasattr(objinfo, "populate_obj"):
#                             asset_obj = bpy.data.objects.get(objinfo.populate_obj)
#                             place_obj = objinfo.obj
#                             if not np.allclose(asset_obj.location, place_obj.location):
#                                 a = 1
#                             asset_obj.rotation_mode = "XYZ"
#                             place_obj.rotation_mode = "XYZ"
#                             if not np.allclose(
#                                 asset_obj.rotation_euler, place_obj.rotation_euler
#                             ):
#                                 a = 1
#                     p.run_stage(
#                         "populate_assets",
#                         populate.populate_state_placeholders_mid,
#                         state,
#                         update_trimesh=False,
#                         use_chance=False,
#                     )

#                     solver.del_no_relation_objects()
#                     # save_path = "debug3.blend"
#                     # bpy.ops.twm.save_as_mainfile(filepath=save_path)
#                     stop = evaluate.del_top_collide_obj(state, iter)

#                     solver.del_no_relation_objects()

#                     if not bpy.app.background:
#                         invisible_others()
#                         bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
#                         visible_others()

#                 for name in list(state.objs.keys())[::-1]:
#                     if name in state.objs.keys():
#                         if name != "newroom_0-0":
#                             if not all_relations_valid(
#                                 state, name, use_initial=True, fix_pos=True
#                             ):
#                                 if check_support(state, name, ratio=0.6):
#                                     # continue if children object is supported > 60%
#                                     continue
#                                 print("all_relations_valid not valid ", name)
#                                 objname = state.objs[name].obj.name
#                                 delete_obj_with_children(
#                                     state.trimesh_scene,
#                                     objname,
#                                     delete_blender=True,
#                                     delete_asset=True,
#                                 )
#                                 state.objs.pop(name)
#                     if not bpy.app.background:
#                         invisible_others(hide_placeholder=True)
#                         bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
#                         visible_others()
#                 solver.del_no_relation_objects()

#         # state,solver = solve_objects.solve_medium_object(stages,limits,solver,state,p,consgraph,overrides)
#         # state,solver = solve_objects.solve_small_object(stages,limits,solver,state,p,consgraph,overrides)
#     record.record_scene(
#         state, solver, terrain, house_bbox, solved_bbox, camera_rigs, iter, p
#     )

#     evaluate.eval_metric(state, iter, remove_bad=True)

#     record_success()

#     # save_path = "debug.blend"
#     # bpy.ops.wm.save_as_mainfile(filepath=save_path)
#     return {
#         "height_offset": height,
#         "whole_bbox": house_bbox,
#     }


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


def cleanup():
    """Cleanup function to properly shutdown socket server"""
    global socket_server
    if socket_server:
        logger.info("Shutting down socket server...")
        socket_server.stop()
        socket_server = None


def stop_server_command():
    """Command to stop the socket server"""
    cleanup()
    return {"status": "server_stopped"}


def main(args):
    global global_overrides, global_configs
    global state, solver, terrain, house_bbox, solved_bbox, camera_rigs, p

    # Initialize basic setup
    scene_seed = init.apply_scene_seed(args.seed)

    # Add terrain disable override by default to avoid landlab dependency issues
    overrides = args.overrides[:]  # Make a copy
    if "compose_indoors.terrain_enabled=False" not in overrides:
        overrides.append("compose_indoors.terrain_enabled=False")

    # # Store overrides and configs globally for use in timer handler
    global_overrides = overrides[:]
    global_configs = args.configs[:]  # Store the gin config files
    logger.info(f"Stored global overrides: {global_overrides}")
    logger.info(f"Stored global configs: {global_configs}")

    init.apply_gin_configs(
        configs=["base_indoors.gin"] + args.configs,
        overrides=overrides,
        config_folders=[
            "infinigen_examples/configs_indoor",
            "infinigen_examples/configs_nature",
        ],
    )
    constants.initialize_constants()

    # Start socket server
    logger.info("Starting socket server for remote commands...")
    start_socket_server()

    # Register Blender timer for action handling
    if not bpy.app.background:
        # Option 1: Use global overrides (current implementation)
        bpy.app.timers.register(blender_action_handler, persistent=True)
        logger.info("Blender action handler registered. Waiting for socket commands...")
        logger.info("Send JSON commands to localhost:12345 with format:")
        iter = args.iter
        
        logger.info('{"iter": '+str(iter)+ ', "save_dir": '+args.save_dir+'}')

    else:
        # If running in background mode, execute once with provided args
        execute_main_logic(args)


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
        "-s", "--seed", default=0, help="The seed used to generate the scene"
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

   
    save_dir = args.save_dir
    os.environ["save_dir"] = save_dir

    # Only load from args.json if it exists (for backward compatibility)
    args_json_path = f"{save_dir}/args.json"
    if os.path.exists(args_json_path):
        try:
            with open(args_json_path, "r") as f:
                j = json.load(f)
                args.iter = j.get("iter", args.iter)
                args.action = j.get("action", args.action)
                args.description = j.get("description", args.description)
                args.inplace = j.get("inplace", args.inplace)
                args.json_name = j.get("json_name", args.json_name)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                f"Could not load args.json: {e}. Using command line arguments."
            )

    # Create necessary directories
    print(save_dir)
    if not os.path.exists(f"{save_dir}/args"):
        os.makedirs(f"{save_dir}/args", exist_ok=True)
        os.makedirs(f"{save_dir}/record_files", exist_ok=True)
        os.makedirs(f"{save_dir}/record_scene", exist_ok=True)

    if args.inplace and os.path.exists(f"{save_dir}/args/args_{args.iter}.json"):
        os.system(
            f"cp {save_dir}/args/args_{args.iter}.json {save_dir}/args/args_{args.iter}_inplaced.json"
        )

    # Save current args
    if os.path.exists(args_json_path):
        os.system(f"cp {args_json_path} {save_dir}/args/args_{args.iter}.json")

    try:
        main(args)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, cleaning up...")
        cleanup()
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        cleanup()
        raise
