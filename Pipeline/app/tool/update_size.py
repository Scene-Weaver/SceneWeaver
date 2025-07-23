import json
import os

from gpt import GPT4

from app.prompt.gpt.update_size import system_prompt, user_prompt
from app.tool.base import BaseTool
from app.tool.update_infinigen import update_infinigen
from app.utils import dict2str, extract_json, lst2str

DESCRIPTION = """
Modify Object Sizes with GPT.
Works with all room types. Best suited for significant size adjustments rather than minor refinements.

Focus primarily on objects placed on a supporting surface.
Use Case 1: Resizing objects with abnormal proportions (e.g., an object on a table that is over one meter tall).
Use Case 2: Scaling objects to meet functional requirements (e.g., enlarging a table when a larger one is needed).

Strengths: Effective at adjusting specific object sizes.
Limitations: Cannot modify overall room dimensions. Should only be used when necessary due to potential scene inconsistencies.

"""


class UpdateSizeExecute(BaseTool):
    name: str = "update_size"
    description: str = DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "ideas": {
                "type": "string",
                "description": "(required) The ideas to adjust size in this step.",
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
            success = update_infinigen(action, iter, json_name, ideas=ideas)
            assert success
            return "Successfully Modify sizes with GPT."
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
