import json
import os

from gpt import GPT4

import app.prompt.gpt.init_gpt as prompts
from app.tool.base import BaseTool
from app.tool.update_infinigen import update_infinigen
from app.utils import dict2str, extract_json, lst2str

DESCRIPTION = """
Using GPT to generate the foundamental scene.

Supported Room Types: any room type.
Use Case 1: Create an accurate and foundational layout.

Strengths: Align well with user demand. More details. Highly versatile and capable of generating scenes for any room type and complex user requirement. Flexible with respect to room design and customization.
Weaknesses: May not be as real as data-driven methods. 

"""


class InitGPTExecute(BaseTool):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "init_gpt"
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
                "description": "(required) The roomtype to load or generate.",
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
            # init scene
            json_name, roomsize = self.gen_gpt_scene(user_demand, ideas, roomtype)
            save_dir = os.getenv("save_dir")
            with open(f"{save_dir}/roominfo.json", "w") as f:
                info = {
                    "action": action,
                    "ideas": ideas,
                    "roomtype": roomtype,
                    "roomsize": roomsize,
                    "save_dir": os.getenv("save_dir"),
                }
                json.dump(info, f, indent=4)
            os.system(
                f"cp  {save_dir}/roominfo.json ../run/roominfo.json"
            )
            success = update_infinigen(action, iter, json_name, ideas=ideas)
            assert success

            return "Successfully initialize scene with GPT."
        except Exception:
            return "Error initializing scene with GPT."

    def gen_gpt_scene(self, user_demand, ideas, roomtype):
        json_name = self.generate_scene_iter0(user_demand, ideas, roomtype)
        with open(json_name, "r") as f:
            j = json.load(f)
        roomsize = j["roomsize"]
        return json_name, roomsize

    def generate_scene_iter0(self, user_demand, ideas, roomtype):
        gpt = GPT4("4.1")

        results = dict()

        ### 1. get big object, count, and relation
        user_prompt = prompts.step_1_big_object_prompt_user.format(
            demand=user_demand, ideas=ideas, roomtype=roomtype
        )
        prompt_payload = gpt.get_payload(
            prompts.step_1_big_object_prompt_system, user_prompt
        )
        gpt_text_response = gpt(payload=prompt_payload, verbose=True)
        print(gpt_text_response)

        # gpt_text_response ='{\n    "Roomtype": "Living Room",\n    "Category list of big object": {\n        "sofa": 2,\n        "armchair": 2,\n        "coffee table": 1,\n        "TV stand": 1,\n        "large shelf": 1,\n        "side table": 2,\n        "floor lamp": 2\n    },\n    "Object against the wall": ["TV stand", "large shelf"],\n    "Relation between big objects": [\n        ["armchair", "coffee table", "front_against"],\n        ["sofa", "coffee table", "front_against"],\n        ["side table", "sofa", "side_by_side"],\n        ["floor lamp", "armchair", "side_by_side"]\n    ]\n}'

        # response = [i for i in gpt_text_response.split("\n") if len(i)>0]
        gpt_dict_response = extract_json(gpt_text_response)
        roomsize = gpt_dict_response["Room size"]
        big_category_dict = gpt_dict_response["Category list of big object"]
        big_category_list = list(big_category_dict.keys())
        category_against_wall = gpt_dict_response["Object against the wall"]
        relation_big_object = gpt_dict_response["Relation between big objects"]

        # # Category list of big objects: [1 checkout counter, 5 bookshelves, 2 reading tables, 8 chairs]
        # # Object against the wall: [bookshelves]
        # # Relation between big objects: [chair, reading table, front_against]

        ##### 5  generate position big
        big_category_dict_str = dict2str(big_category_dict)
        category_against_wall_str = lst2str(category_against_wall)
        relation_big_object_str = lst2str(relation_big_object)
        roomsize_str = lst2str(roomsize)

        user_prompt = prompts.step_5_position_prompt_user.format(
            big_category_dict=big_category_dict_str,
            category_against_wall=category_against_wall_str,
            relation_big_object=relation_big_object_str,
            demand=user_demand,
            roomsize=roomsize_str,
        )
        prompt_payload = gpt.get_payload(
            prompts.step_5_position_prompt_system, user_prompt
        )
        success = False
        iter = 0
        while not success and iter < 5:
            iter += 1
            gpt_text_response = gpt(payload=prompt_payload, verbose=True)
            print(gpt_text_response)

            # gpt_text_response = '{\n    "Roomtype": "Bookstore",\n    "list of given category names": ["sofa", "armchair", "coffee table", "TV stand", "large shelf", "side table", "floor lamp", "remote control", "book", "magazine", "decorative bowl", "photo frame", "vase", "candle", "coaster", "plant"],\n    "Mapping results": {\n        "sofa": "seating.SofaFactory",\n        "armchair": "seating.ArmChairFactory",\n        "coffee table": "tables.CoffeeTableFactory",\n        "TV stand": "shelves.TVStandFactory",\n        "large shelf": "shelves.LargeShelfFactory",\n        "side table": "tables.SideTableFactory",\n        "floor lamp": "lamp.FloorLampFactory",\n        "remote control": null,\n        "book": "table_decorations.BookStackFactory",\n        "magazine": null,\n        "decorative bowl": "tableware.BowlFactory",\n        "photo frame": null,\n        "vase": "table_decorations.VaseFactory",\n        "candle": null,\n        "coaster": null,\n        "plant": "tableware.PlantContainerFactory"\n    }\n}'
            try:
                gpt_dict_response = extract_json(
                    gpt_text_response.replace("'", '"').replace("None", "null")
                )
                success = True
            except:
                success = False
        Placement_big = gpt_dict_response["Placement"]

        small_category_list = []
        relation_small_object = []
        Placement_small = []

        # #### 3. get object class name in infinigen
        category_list = big_category_list + small_category_list
        s = lst2str(category_list)

        user_prompt = prompts.step_3_class_name_prompt_user.format(
            category_list=s, demand=user_demand
        )
        system_prompt = prompts.step_3_class_name_prompt_system

        prompt_payload = gpt.get_payload(system_prompt, user_prompt)
        gpt_text_response = gpt(payload=prompt_payload, verbose=True)
        print(gpt_text_response)

        gpt_dict_response = extract_json(
            gpt_text_response.replace("'", '"').replace("None", "null")
        )
        name_mapping = gpt_dict_response["Mapping results"]

        results["user_demand"] = user_demand
        results["roomsize"] = roomsize
        results["big_category_dict"] = big_category_dict
        results["category_against_wall"] = category_against_wall
        results["relation_big_object"] = relation_big_object
        results["small_category_list"] = small_category_list
        results["relation_small_object"] = relation_small_object
        results["name_mapping"] = name_mapping
        results["gpt_text_response"] = gpt_text_response
        results["Placement_big"] = Placement_big
        results["Placement_small"] = Placement_small

        save_dir = os.getenv("save_dir")
        json_name = f"{save_dir}/pipeline/init_gpt_results_{iter}.json"
        with open(json_name, "w") as f:
            json.dump(results, f, indent=4)

        return json_name
