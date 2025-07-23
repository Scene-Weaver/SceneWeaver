import json
import os

from gpt import GPT4

from app.prompt.gpt.remove_obj import example, system_prompt, user_prompt
from app.tool.base import BaseTool
from app.tool.update_infinigen import update_infinigen
from app.utils import dict2str, extract_json, lst2str

DESCRIPTION = """
Remove objects with GPT. Works with all room types.

Use Case 1: Remove redundant and unnecessary objects when the scene is crowded or when there are too many objects. (e.g., eliminate a table in the corner)
Use Case 2: Remove objects that does not belongs to this roomtype. (e.g., eliminate the bed in the dining room)
Use Case 3: Remove objects when the collision/outside problem has not been solved for several attempts by other tools. (e.g., eliminate the object outside the room)
Use Case 3: Remove small objects (usually with collision or outside the supporting surface) when their supporter or container has no enough space to support them. (e.g., eliminate some small objects or  when the nightstand is overloaded)

Strengths: Excels at removing specific objects. Can solve collison and crowded problems directly. 
Weaknesses: Can not add objects or replace objects. You must use this method carefully to avoid mistaken deletion.
"""


class RemoveExecute(BaseTool):
    name: str = "remove_object"
    description: str = DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "ideas": {
                "type": "string",
                "description": "(required) The ideas to remove objects in this step.",
            },
        },
        "required": ["ideas"],
    }

    def execute(self, ideas: str) -> str:
        user_demand = os.getenv("UserDemand")
        iter = int(os.getenv("iter"))
        roomtype = os.getenv("roomtype")
        action = self.name
        try:
            # find scene
            json_name = self.update_scene_gpt(user_demand, ideas, iter, roomtype)
            # json_name = update_ds(user_demand,ideas,iter,roomtype)

            success = update_infinigen("remove_object", iter, json_name, ideas=ideas)
            assert success
            return "Successfully remove object with GPT."
        except Exception:
            return "Error remove object with GPT"

    def update_scene_gpt(self, user_demand, ideas, iter, roomtype):
        save_dir = os.getenv("save_dir")
        render_path = f"{save_dir}/record_scene/render_{iter-1}.jpg"
        with open(f"{save_dir}/record_scene/layout_{iter-1}.json", "r") as f:
            layout = json.load(f)

        roomsize = layout["roomsize"]
        roomsize = lst2str(roomsize)

        structure = dict2str(layout["structure"])
        layout = dict2str(layout["objects"])

        system_prompt_1 = system_prompt
        user_prompt_1 = user_prompt.format(
            roomtype=roomtype,
            roomsize=roomsize,
            layout=layout,
            structure=structure,
            user_demand=user_demand,
            ideas=ideas,
            example=example,
        )

        gpt = GPT4(version="4.1")

        prompt_payload = gpt.get_payload_scene_image(
            system_prompt_1, user_prompt_1, render_path=render_path
        )
        gpt_text_response = gpt(payload=prompt_payload, verbose=True)
        print(gpt_text_response)

        json_name = f"{save_dir}/pipeline/remove_obj_results_{iter}_response.json"
        with open(json_name, "w") as f:
            json.dump(gpt_text_response, f, indent=4)

        new_layout = extract_json(gpt_text_response)

        json_name = f"{save_dir}/pipeline/remove_obj_results_{iter}.json"
        with open(json_name, "w") as f:
            json.dump(new_layout, f, indent=4)

        return json_name
