import json
import os

from gpt import GPT4

from app.prompt.gpt.update_gpt import system_prompt, user_prompt
from app.tool.base import BaseTool
from app.tool.update_infinigen import update_infinigen
from app.utils import dict2str, extract_json, lst2str

DESCRIPTION = """
Modify layout with GPT. Works with all room types. 
This method is not recommended for slight layout adjustments; it is better suited for major changes when necessary.

Use Case 1: Adjusting objects' placement when the objects is not well-placed.
Use Case 2: Change objects' scale when the size does not match the requirement.

Strengths: Excels at modifying specific objects.
Weaknesses:  **Can not add objects**. Can not solve all the problem when the room is crowded. Bad in modify rotation. May lack precision and occasionally overlook details. 
Can not obey the current relation, such as move object away from the wall when the object is against wall.
"""


class UpdateLayoutExecute(BaseTool):
    name: str = "update_layout"
    description: str = DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "ideas": {
                "type": "string",
                "description": "(required) The ideas to adjust layout or remove objects in this step.",
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
            success = update_infinigen("update", iter, json_name, ideas=ideas)
            assert success
            return "Successfully Modify layout with GPT."
        except Exception:
            return "Error Modify layout with GPT"

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
        )

        gpt = GPT4(version="4.1")

        prompt_payload = gpt.get_payload_scene_image(
            system_prompt_1, user_prompt_1, render_path=render_path
        )
        gpt_text_response = gpt(payload=prompt_payload, verbose=True)
        print(gpt_text_response)

        # json_name = f"{save_dir}/pipeline/update_gpt_results_{iter}_response.json"
        # with open(json_name, "w") as f:
        #     json.dump(gpt_text_response, f, indent=4)

        new_layout = extract_json(gpt_text_response)

        json_name = f"{save_dir}/pipeline/update_gpt_results_{iter}.json"
        with open(json_name, "w") as f:
            json.dump(new_layout, f, indent=4)

        return json_name
