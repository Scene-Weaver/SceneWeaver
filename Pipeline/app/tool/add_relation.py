import json
import os

from gpt import GPT4

from app.prompt.add_relation import example, system_prompt, user_prompt
from app.tool.base import BaseTool
from app.tool.update_infinigen import update_infinigen
from app.utils import dict2str, extract_json, lst2str

DESCRIPTION = """
Add explicit relation between objects in the current scene according to the layout. 
Sometimes the relation is encoded in the layout coordinate rather than represented explicitly, making it difficult to manage.
Explicit relations will make the scene more tidy.

Use Case 1: Add explicit relation between large objects, according to the layout, to make the scene better-organized. 
Use Case 2: Add new relation between large objects, make the scene better-organized. 
Use Case 3: Add againts_wall relation to large objects, make the objects stand against wall. 
Use Case 4: Add floating small objects on/in a large object. 

**Note:** Each object can have only one parent object (except for the room). Do not add relation between small objects.

The optional relations between objects are: 
1.front_against: child_obj's front faces to parent_obj, and stand very close.
2.front_to_front: child_obj's front faces to parent_obj's front, and stand very close.
3.leftright_leftright: child_obj's left or right faces to parent_obj's left or right, and stand very close. 
4.side_by_side: child_obj's side(left, right , or front) faces to parent_obj's side(left, right , or front), and stand very close. 
5.back_to_back: child_obj's back faces to parent_obj's back, and stand very close. 
6.ontop: child_obj is placed on the top of parent_obj.
7.on: child_obj is placed on the top of or inside parent_obj.

The optional relations between object and room are: 
1.against_wall: child_obj's back faces to the wall of the room, and stand very close.
2.side_against_wall: child_obj's side(left, right , or front) faces to the wall of the room, and stand very close.
3.on_floor: child_obj stand on the parent_obj, which is the floor of the room.

Strengths: Can add relation between objects, make the scene tidy and well-organized quickly. 
Weaknesses: Can not fix the layout problem, such as placing the object into the right place accurately. 

"""


class AddRelationExecute(BaseTool):
    name: str = "add_relation"
    description: str = DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "ideas": {
                "type": "string",
                "description": "(required) The ideas to add relations in this step.",
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
            json_name = add_relation(user_demand, ideas, iter, roomtype)
            success = update_infinigen(
                action, iter, json_name, inplace=False, invisible=True, ideas=ideas
            )
            assert success
            return "Successfully add relation between objects."
        except Exception:
            return "Error adding relation between objects"


def add_relation(user_demand, ideas, iter, roomtype):
    save_dir = os.getenv("save_dir")
    if iter == 0:
        render_path = f"{save_dir}/record_scene/render_{iter}.jpg"
        with open(f"{save_dir}/record_scene/layout_{iter}.json", "r") as f:
            layout = json.load(f)
    else:
        render_path = f"{save_dir}/record_scene/render_{iter-1}.jpg"
        with open(f"{save_dir}/record_scene/layout_{iter-1}.json", "r") as f:
            layout = json.load(f)

    roomsize = layout["roomsize"]
    roomsize = lst2str(roomsize)

    structure = dict2str(layout["structure"])
    layout = dict2str(layout["objects"])

    system_prompt_1 = system_prompt
    user_prompt_1 = (
        user_prompt.format(
            roomtype=roomtype,
            roomsize=roomsize,
            layout=layout,
            structure=structure,
            user_demand=user_demand,
            ideas=ideas,
        )
        + example
    )

    gpt = GPT4(version="4.1")

    prompt_payload = gpt.get_payload_scene_image(
        system_prompt_1, user_prompt_1, render_path=render_path
    )
    gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    print(gpt_text_response)

    json_name = f"{save_dir}/pipeline/add_relation_results_{iter}_response.json"
    with open(json_name, "w") as f:
        json.dump(gpt_text_response, f, indent=4)

    new_layout = extract_json(gpt_text_response)

    json_name = f"{save_dir}/pipeline/add_relation_results_{iter}.json"
    with open(json_name, "w") as f:
        json.dump(new_layout, f, indent=4)

    return json_name
