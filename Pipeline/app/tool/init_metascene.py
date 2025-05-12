import json
import multiprocessing
import os
import random
import sys
from io import StringIO
from typing import Dict

from gpt import GPT4

from app.tool.add_relation import add_relation
from app.tool.base import BaseTool
from app.tool.metascene_frontview import get_scene_frontview
from app.tool.update_infinigen import update_infinigen
from app.utils import dict2str, extract_json

DESCRIPTION = """
Load the most related scene from a Real2Sim indoor scene dataset as the basic scene.
Ideal for generating foundational layouts for common room types.

Only Supported Room Types: living room, dining room, bedroom, bathroom, kitchen, office, laundry room, and classroom.
Use Case 1: Create a foundational layout.

Strengths: Provides a ready-made layout based on real-world data. Rich of details.
Weaknesses: Fixed layout, need to modify with other methods to meet user demand. 
"""
# The layout is in low quality. Assets' quality is unstable.

class InitMetaSceneExecute(BaseTool):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "init_metascene"
    description: str = DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "ideas": {
                "type": "string",
                "description": "(required) The idea to init the scene in this step.",
            },
            "roomtype": {
                "type": "string",
                "description": "(required) The roomtype to load.",
            },
        },
        "required": ["ideas", "roomtype"],
    }

    def execute(self, ideas: str, roomtype: str) -> str:
        """
        Save content to a file at the specified path.

        Args:
            content (str): The content to save to the file.
            file_path (str): The path where the file should be saved.
            mode (str, optional): The file opening mode. Default is 'w' for write. Use 'a' for append.

        Returns:
            str: A message indicating the result of the operation.
        """
        user_demand = os.getenv("UserDemand")
        iter = int(os.getenv("iter"))
        os.environ["roomtype"] = roomtype

        action = self.name
        try:
            # # find scene
            save_dir = os.getenv("save_dir")
            json_name, roomsize = self.find_metascene(user_demand, ideas, roomtype)
            roomsize = self.get_roomsize(user_demand, ideas, roomsize, roomtype)
            success = get_scene_frontview(json_name)
            with open("/home/yandan/workspace/infinigen/roominfo.json", "w") as f:
                info = {
                    "action": action,
                    "ideas": ideas,
                    "roomtype": roomtype,
                    "roomsize": roomsize,
                    "scene_id": json_name,
                    "save_dir": save_dir,
                }
                json.dump(info, f, indent=4)
            os.system(
                f"cp /home/yandan/workspace/infinigen/roominfo.json {save_dir}/roominfo.json"
            )
            success = update_infinigen(action, iter, json_name, ideas=ideas)
            assert success

            # add relation
            action = "add_relation"
            json_name = add_relation(user_demand, ideas, iter, roomtype)
            success = update_infinigen(
                action, iter, json_name, inplace=True, invisible=True, ideas=ideas
            )
            assert success

            return f"Successfully initialize scene by loading MetaScene."
        except Exception as e:
            return f"Error initializing scene by loading MetaScene."

    def find_metascene(self, user_demand, ideas, roomtype):
        def statistic_obj_nums(scene_id):
            filename = f"/mnt/fillipo/huangyue/recon_sim/7_anno_v4/export_stage2_sm/{scene_id}/metadata.json"
            with open(filename, "r") as f:
                data = json.load(f)
            category_count = {}
            for key, value in data.items():
                if value in ["floor", "wall", "window", "ceiling"]:
                    continue
                category_count[value] = category_count.get(value, 0) + 1

            return category_count

        def find_scene_id():
            scene_ids = list(scenes.keys())
            scene_id_cands = []
            random.shuffle(scene_ids)
            for scene_id in scene_ids:
                try:
                    scene_type = scenes[scene_id]["roomtype"]
                    if len(scene_type) > 1:
                        continue
                    for info in scene_type:
                        if roomtype in info["predicted"] and info["confidence"] > 0.8:
                            scene_id_cands.append(scene_id)
                            break
                except:
                    a = 1
            category_counts = dict()
            for scene_id in scene_id_cands:
                category_count = statistic_obj_nums(scene_id)
                category_counts[scene_id] = category_count

            with open("category_counts.json", "w") as f:
                json.dump(category_counts, f, indent=4)

            scene_id = self.match_scene_id(
                category_counts, user_demand, ideas, roomtype
            )

            return scene_id

        if roomtype.endswith("room"):
            roomtype = roomtype[:-4].strip()
        basedir = "/mnt/fillipo/yandan/metascene/export_stage2_sm"

        with open(f"{basedir}/statistic.json", "r") as f:
            j = json.load(f)

        scenes = j["scenes"]

        scene_id = find_scene_id()
        # scene_id = "scene0592_00"
        json_name = scene_id

        with open(
            "/mnt/fillipo/yandan/metascene/export_stage2_sm/roomsize.json", "r"
        ) as f:
            data = json.load(f)
            room_size = data[scene_id]
            room_size = [round(room_size["size_x"], 1), round(room_size["size_y"], 1)]

        return json_name, room_size

    def match_scene_id(self, category_counts, user_demand, ideas, roomtype):
        from app.prompt.metascene.match_sceneid import system_prompt, user_prompt

        category_counts = dict2str(category_counts)

        user_prompt_1 = user_prompt.format(
            user_demand=user_demand,
            roomtype=roomtype,
            ideas=ideas,
            category_counts=category_counts,
        )

        gpt = GPT4(version="4.1")

        prompt_payload = gpt.get_payload(system_prompt, user_prompt_1)
        gpt_text_response = gpt(payload=prompt_payload, verbose=True)
        print("matched scene id :", gpt_text_response)

        return gpt_text_response

    def get_roomsize(self, user_demand, ideas, roomsize, roomtype):
        from app.prompt.metascene.get_roomsize import system_prompt, user_prompt

        user_prompt_1 = user_prompt.format(
            user_demand=user_demand, roomtype=roomtype, ideas=ideas, roomsize=roomsize
        )

        gpt = GPT4(version="4o")

        prompt_payload = gpt.get_payload(system_prompt, user_prompt_1)
        gpt_text_response = gpt(payload=prompt_payload, verbose=True)
        print("roomsize :", gpt_text_response)
        roomsize = gpt_text_response.split(",")
        roomsize = [round(float(roomsize[0]), 1), round(float(roomsize[1]), 1)]

        return roomsize
