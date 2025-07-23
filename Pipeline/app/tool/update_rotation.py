import json
import os

from gpt import GPT4

from app.prompt.gpt.update_rotation import system_prompt, user_prompt
from app.tool.base import BaseTool
from app.tool.update_infinigen import update_infinigen
from app.utils import dict2str, extract_json, lst2str

DESCRIPTION = """
Adjust object rotations with GPT to optimize room layout. Suitable for all room types. 

Use Case 1: Fix incorrect object orientations, such as a bed facing the wall or a chair turned away from a desk.
Use Case 2: Improve spatial organization by aligning objects more naturally with the room structure or usage context (e.g., rotate a sofa to face a TV or a chair to face a table).

Strengths: Helps improve the visual and functional coherence of a room. Can automatically identify misaligned items and suggest better orientations based on typical room usage.
Weaknesses: Does not move or add/remove objects. Only focus on rotation 
"""


class UpdateRotationExecute(BaseTool):
    name: str = "update_rotation"
    description: str = DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "ideas": {
                "type": "string",
                "description": "(required) The ideas to adjust rotation in this step.",
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
            return "Successfully Modify rotation with GPT."
        except Exception:
            return "Error Modify rotation with GPT"

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

        # json_name = f"{save_dir}/pipeline/update_rotation_results_{iter}_response.json"
        # with open(json_name, "w") as f:
        #     json.dump(gpt_text_response, f, indent=4)

        new_layout = extract_json(gpt_text_response)

        json_name = f"{save_dir}/pipeline/update_rotation_results_{iter}.json"
        with open(json_name, "w") as f:
            json.dump(new_layout, f, indent=4)

        return json_name
