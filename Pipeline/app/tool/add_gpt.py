import json
import os

from gpt import GPT4

import app.prompt.gpt.add_gpt as prompts1
import app.prompt.gpt.init_gpt as prompts0
from app.tool.init_gpt import InitGPTExecute
from app.tool.update_infinigen import update_infinigen
from app.utils import extract_json, lst2str

DESCRIPTION = """
Using GPT to add additional objects into the current scene.

Use Case 1: Add large objects in the current scene.
Use Case 2: Add 1-2 small objects on the top of small supporting furniture, such as nightstand and cabinet,  when there is enough space. (e.g., add a cup on the nightstand).
Use Case 3: Add several small objects on the top of large supporting furniture, such as dining table and desk, when there is enough space. (e.g., add daily tableware on the dining table).
Use Case 4: Add several small objects inside the large furniture. (e.g., add books in the shelf).
Use Case 5: Add functional objects or decorations on the wall. (e.g., add painting, mirror, and TV on the wall).

You **MUST** not :
1.Do not add objects where there is no available space.
2.Do not add small objects on any tall furniture, such as wardrob.

Strengths: The location is accurate. Can add objects inside a container, such as objects in the shelf.
Weaknesses: **Can not add objects on the ceiling.** The rotation of asset is not accurate. Relation between small objects is not accurate. Can not modify objects in the current scene.  

"""


class AddGPTExecute(InitGPTExecute):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "add_gpt"
    description: str = DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "ideas": {
                "type": "string",
                "description": "(required) The ideas to add objects in this step.",
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
            json_name = self.add_gpt(user_demand, ideas, iter, roomtype)

            success = update_infinigen(action, iter, json_name, ideas=ideas)
            assert success

            return "Successfully add objects with GPT."
        except Exception:
            return "Error adding objects with GPT"

    def add_gpt(self, user_demand, ideas, iter, roomtype):
        json_name = self.generate_scene_iter1_gpt(user_demand, ideas, iter, roomtype)

        return json_name

    def generate_scene_iter1_gpt(self, user_demand, ideas, iter, roomtype):
        gpt = GPT4(version="4.1")

        results = dict()
        save_dir = os.getenv("save_dir")
        render_path = f"{save_dir}/record_scene/render_{iter-1}.jpg"
        with open(f"{save_dir}/record_scene/layout_{iter-1}.json", "r") as f:
            layout = json.load(f)

        roomsize = layout["roomsize"]

        roomsize_str = f"[{roomsize[0]},{roomsize[1]}]"
        step_1_big_object_prompt_user = prompts1.step_1_big_object_prompt_user.format(
            demand=user_demand,
            roomtype=roomtype,
            ideas=ideas,
            roomsize=roomsize_str,
            scene_layout=layout["objects"],
            structure=layout["structure"],
        )

        prompt_payload = gpt.get_payload_scene_image(
            prompts1.step_1_big_object_prompt_system,
            step_1_big_object_prompt_user,
            render_path,
        )
        gpt_text_response = gpt(payload=prompt_payload, verbose=True)
        print(gpt_text_response)

        gpt_dict_response = extract_json(gpt_text_response)
        results = gpt_dict_response

        # #### 2. get object class name in infinigen
        category_list = gpt_dict_response["Number of new furniture"]
        if len(category_list.keys()) == 0:
            return "Nothing"
        s = lst2str(list(category_list.keys()))
        user_prompt = prompts0.step_3_class_name_prompt_user.format(
            category_list=s, demand=user_demand
        )
        system_prompt = prompts0.step_3_class_name_prompt_system
        prompt_payload = gpt.get_payload(system_prompt, user_prompt)
        gpt_text_response = gpt(payload=prompt_payload, verbose=True)
        print(gpt_text_response)

        gpt_dict_response = extract_json(
            gpt_text_response.replace("'", '"').replace("None", "null")
        )
        name_mapping = gpt_dict_response["Mapping results"]
        results["name_mapping"] = name_mapping

        save_dir = os.getenv("save_dir")
        json_name = f"{save_dir}/pipeline/add_gpt_results_{iter}.json"
        with open(json_name, "w") as f:
            json.dump(results, f, indent=4)
        return json_name
