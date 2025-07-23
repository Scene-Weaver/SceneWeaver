import json
import os
import random

import numpy as np

from app.tool.add_relation import add_relation
from app.tool.base import BaseTool
from app.tool.get_roomsize import get_roomsize
from app.tool.update_infinigen import update_infinigen

DESCRIPTION = """
Using neural network to generate a scene as the basic scene.
The neural network is trained on the 3D Front indoor dataset.

Supported Room Types: Living room, bedroom, and dining room.
Use Case 1: Create a foundational layout.

Strengths: Room is clean and tidy. Assets in good quality.
Weaknesses: Fixed layout, less details. Need to modify with other methods to meet user demand.
"""


class InitPhySceneExecute(BaseTool):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "init_physcene"
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
            # #find scene
            save_dir = os.getenv("save_dir")
            json_name, roomsize = self.find_physcene(user_demand, ideas, roomtype)
            roomsize = get_roomsize(user_demand, ideas, roomsize, roomtype)

            with open(f"{save_dir}/roominfo.json", "w") as f:
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
                f"cp {save_dir}/roominfo.json ~/workspace/SceneWeaver/roominfo.json"
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

            return "Successfully generate a scene by neural network."
        except Exception:
            return "Error generating a scene by neural network."

    def find_physcene(self, user_demand, ideas, roomtype):
        roomtype = roomtype.lower()
        if roomtype.endswith("room"):
            roomtype = roomtype[:-4].strip()
        basedir = "~/workspace/PhyScene/3D_front/generate_filterGPN_clean/"
        files = os.listdir(basedir)
        random.shuffle(files)
        for filename in files:
            if filename.endswith(".json") and roomtype in filename.lower():
                break

        json_name = f"{basedir}/{filename}"

        def calculate_room_size(data):
            min_coords = np.array([float("inf"), float("inf"), float("inf")])
            max_coords = np.array([-float("inf"), -float("inf"), -float("inf")])

            for objects in data["ThreedFront"].values():
                for obj in objects:
                    position = np.array(obj["position"])  # Object's position
                    size = np.array(
                        obj["size"]
                    )  # Half-size for bounding box calculation

                    # Compute object's bounding box min and max coordinates
                    obj_min = position - size
                    obj_max = position + size

                    # Update overall min/max coordinates
                    min_coords = np.minimum(min_coords, obj_min)
                    max_coords = np.maximum(max_coords, obj_max)

            # Calculate room size (max - min)
            room_size = 2 * np.maximum(abs(max_coords), abs(min_coords))
            return room_size[0], room_size[2]

        with open(json_name, "r") as f:
            data = json.load(f)
            room_size = calculate_room_size(data)

        return json_name, room_size
