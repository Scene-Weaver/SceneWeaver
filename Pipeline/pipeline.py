import json
import os
import random

import numpy as np
from add_deepseek import generate_scene_iter1_ds
from add_gpt import generate_scene_iter1_gpt
from add_relation_gpt import add_relation
from gen_acdc_candidates import gen_ACDC_cand
from gen_SD_prompt import gen_SD_prompt
from get_action import get_action0, get_action1
from get_action_ds import get_action0_ds, get_action1_ds
from get_roomsize import get_roomsize
from init_gpt import generate_scene_iter0
from match_scene import match_scene_id
from update_ds import update_scene_ds
from update_gpt import update_scene_gpt
from evaluation import eval_scene
from metascene_frontview import get_scene_frontview

def get_action(user_demand, iter, ds=False):
    if iter == 0:
        action, ideas, roomtype = get_action0(user_demand, iter)
    else:
        if ds:
            action, ideas, roomtype = get_action1_ds(user_demand, iter)
        else:
            action, ideas, roomtype = get_action1(user_demand, iter)

    return action, ideas, roomtype


def find_physcene(user_demand, ideas, roomtype):
    roomtype = roomtype[:-5]
    basedir = "/home/yandan/workspace/PhyScene/3D_front/generate_filterGPN_clean/"
    files = os.listdir(basedir)
    random.shuffle(files)
    for filename in files:
        filename = "LivingDiningRoom-37001_diningroom.json"
        if filename.endswith(".json") and roomtype in filename.lower():
            break

    json_name = f"{basedir}/{filename}"

    def calculate_room_size(data):
        min_coords = np.array([float("inf"), float("inf"), float("inf")])
        max_coords = np.array([-float("inf"), -float("inf"), -float("inf")])

        for objects in data["ThreedFront"].values():
            for obj in objects:
                position = np.array(obj["position"])  # Object's position
                size = np.array(obj["size"])  # Half-size for bounding box calculation

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


def find_metascene(user_demand, ideas, roomtype):
    # def find_scene_id():
    #     scene_ids = list(scenes.keys())
    #     random.shuffle(scene_ids)
    #     HasFind = False
    #     for scene_id in scene_ids:
    #         scene_type = scenes[scene_id]["roomtype"]
    #         for info in scene_type:
    #             if roomtype in info["predicted"] and info["confidence"]>0.8:
    #                 HasFind = True
    #                 break
    #         if HasFind:
    #             break
    #     return scene_id

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

        scene_id = match_scene_id(category_counts, user_demand, ideas, roomtype)

        return scene_id

    if roomtype.endswith("room"):
        roomtype = roomtype[:-4].strip()
    basedir = "/mnt/fillipo/yandan/metascene/export_stage2_sm"

    with open(f"{basedir}/statistic.json", "r") as f:
        j = json.load(f)

    scenes = j["scenes"]

    scene_id = find_scene_id()
    # scene_id = "scene0035_00"
    json_name = scene_id

    with open("/mnt/fillipo/yandan/metascene/export_stage2_sm/roomsize.json", "r") as f:
        data = json.load(f)
        room_size = data[scene_id]
        room_size = [round(room_size["size_x"], 1), round(room_size["size_y"], 1)]

    return json_name, room_size


def gen_gpt_scene(user_demand, ideas, roomtype):
    json_name = generate_scene_iter0(user_demand, ideas, roomtype)
    with open(json_name, "r") as f:
        j = json.load(f)
    roomsize = j["roomsize"]
    return json_name, roomsize


def add_gpt(user_demand, ideas, iter):
    # json_name = generate_scene_iter1_ds(user_demand,ideas,iter)
    try:
        json_name = generate_scene_iter1_gpt(user_demand, ideas, iter)
    except:
        json_name = generate_scene_iter1_gpt(user_demand, ideas, iter)
    return json_name


def prepare_acdc(user_demand, ideas, roomtype, iter):
    result = gen_ACDC_cand(user_demand, ideas, roomtype, iter)

    return result


def gen_img_SD(SD_prompt, obj_id, obj_size):
    # objtype = obj_id.split("_")[1:]
    # objtype = "_".join(objtype)
    # SD_prompt = gen_SD_prompt(prompt,objtype,obj_size)
    img_filename = "/home/yandan/workspace/infinigen/Pipeline/record/SD_img.jpg"
    j = {"prompt": SD_prompt, "img_savedir": img_filename}
    with open("/home/yandan/workspace/sd3.5/prompt.json", "w") as f:
        json.dump(j, f, indent=4)

    basedir = "/home/yandan/workspace/sd3.5"
    os.system(f"bash {basedir}/run.sh")

    return img_filename


def update_infinigen(action, iter, json_name, description=None, inplace=False,invisible=False):
    j = {
        "iter": iter,
        "action": action,
        "json_name": json_name,
        #  "roomsize": roomsize,
        "description": description,
        "inplace": inplace,
    }

    with open(f"/home/yandan/workspace/infinigen/args.json", "w") as f:
        json.dump(j, f, indent=4)

    if invisible:
        os.system("bash -i /home/yandan/workspace/infinigen/run_invisible.sh")
    else:
        os.system("bash -i /home/yandan/workspace/infinigen/run.sh")

    return


def acdc(img_filename, obj_id, category):
    # objtype = obj_id.split("_")[1:]
    # objtype = "_".join(objtype)
    j = {
        "obj_id": obj_id,
        "objtype": category,
        "img_filename": img_filename,
        "success": False,
        "error": "Unknown",
    }
    with open("/home/yandan/workspace/digital-cousins/args.json", "w") as f:
        json.dump(j, f, indent=4)

    os.system("bash -i /home/yandan/workspace/digital-cousins/run.sh")
    json_name = "/home/yandan/workspace/infinigen/Pipeline/record/acdc_output/step_3_output/scene_0/scene_0_info.json"

    return json_name


def add_init_relation(user_demand, ideas, roomtype):
    json_name = add_relation(user_demand, ideas, iter, roomtype)
    return json_name


def update_gpt(user_demand, ideas, iter, roomtype):
    json_name = update_scene_gpt(user_demand, ideas, iter, roomtype)
    return json_name


def update_ds(user_demand, ideas, iter, roomtype):
    json_name = update_scene_ds(user_demand, ideas, iter, roomtype)
    return json_name


iter = 14
# user_demand = "An office room for 8 people."
user_demand = "You must design a scene iteratively using the tools I designed, it must have one large table with eight chairs placing properly next to the table with appropriate size and scale. You can choose to modify the scene by adding and eliminating objects. It should have a large table with comfortable seating for the family and guests."
# user_demand = "Could you please design a Dining Room for me? Designed for meal gatherings, it should have a large table with comfortable seating for the family and guests, along with ambient lighting to enhance the dining experience."  # "A classic Chinese dining room."

while iter < 15:
    if iter == 0:
        action, ideas, roomtype = get_action(user_demand, iter)
        # action = "init_metascene"
        # ideas = "Add basic dining room layout with a large table and eight chairs.",
        # roomtype = 'dining room'
        # action='init_gpt'
        # ideas='Create a foundational layout for an dining room.'
        
        if action == "init_physcene":
            json_name, roomsize = find_physcene(user_demand, ideas, roomtype)
            roomsize = get_roomsize(user_demand, ideas, roomsize, roomtype)
        elif action == "init_metascene":
            # json_name = "scene0004_00"
            # roomsize=[1.8,9.0]
            json_name, roomsize = find_metascene(user_demand, ideas, roomtype)
            roomsize = get_roomsize(user_demand, ideas, roomsize, roomtype)
            success = get_scene_frontview(json_name)
            if not success:
                a = 1
        elif action == "init_gpt":
            json_name, roomsize = gen_gpt_scene(user_demand, ideas, roomtype)
            # json_name='/home/yandan/workspace/infinigen/Pipeline/record/init_gpt_results.json'
            # roomsize=[5,7]
        else:
            raise ValueError(f"Action is wrong: {action}")

        with open("/home/yandan/workspace/infinigen/roominfo.json", "w") as f:
            info = {
                "action": action,
                "ideas": ideas,
                "roomtype": roomtype,
                "roomsize": roomsize,
            }
            json.dump(info, f, indent=4)
        update_infinigen(action, iter, json_name)
        if action == "init_physcene" or action == "init_metascene":
            json_name = add_init_relation(user_demand, ideas, roomtype)
            update_infinigen("add_relation", iter, json_name, inplace=True,invisible=True)
    else:
        action = None
        while action is None:
            action, ideas, roomtype = get_action(user_demand, iter)  # ,ds=True)
        # action = "add_acdc"
        # ideas = 'Add a lamp on the nightstand, books on the shelf, and a plant on the desk.'
        # roomtype = 'bedroom'

        if action == "add_gpt":
            json_name = add_gpt(user_demand, ideas, iter)
            update_infinigen(action, iter, json_name)

        elif action == "add_acdc":
            steps = prepare_acdc(user_demand, ideas, roomtype, iter)
            # steps = {'9563833_nightstand': {'prompt for SD': 'An entire 54cm * 57cm * 59cm nightstand with a lamp on it.', 'obj category': 'nightstand', 'obj_id': '9563833_nightstand', 'obj_size': [...]}, '2552790_shelf': {'prompt for SD': 'An entire 74cm * 31cm * 73cm shelf with books on it.', 'obj category': 'shelf', 'obj_id': '2552790_shelf', 'obj_size': [...]}, '1133652_desk': {'prompt for SD': 'An entire 115cm * 57cm * 77cm desk with a plant on it.', 'obj category': 'desk', 'obj_id': '1133652_desk', 'obj_size': [...]}}
            # last_prompt = 'A 120cm * 60cm * 70cm simple desk with a monitor, desk lamp, stack of documents, and a coffee mug.'
            # last_img_filename = "/home/yandan/workspace/infinigen/Pipeline/record/SD_img.jpg"
            # last_json_name = "/home/yandan/workspace/infinigen/Pipeline/record/acdc_output/step_3_output/scene_0/scene_0_info.json"
            last_prompt = None
            last_img_filename = None
            last_json_name = None
            inplace = False
            for obj_id, info in steps.items():
                if last_prompt is None or info["prompt for SD"] != last_prompt:
                    update_infinigen("export_supporter",iter,json_name="",description=obj_id)
                    cnt = 0
                    while True and cnt < 5:
                        cnt += 1
                        print(info["prompt for SD"])
                        img_filename = gen_img_SD(
                            info["prompt for SD"], obj_id, info["obj_size"]
                        )  # execute until satisfy the requirement
                        # img_filename = '/home/yandan/workspace/infinigen/Pipeline/record/SD_img.jpg'

                        json_name = acdc(img_filename, obj_id, info["obj category"])

                        with open(
                            "/home/yandan/workspace/digital-cousins/args.json", "r"
                        ) as f:
                            j = json.load(f)
                            if j["success"]:
                                break

                else:
                    img_filename = last_img_filename
                    json_name = last_json_name

                last_prompt = info["prompt for SD"]
                last_img_filename = img_filename
                last_json_name = json_name

                update_infinigen(
                    action, iter, json_name, description=obj_id, inplace=inplace
                )
                inplace = True

        elif action == "update":
            json_name = update_gpt(user_demand, ideas, iter, roomtype)
            # json_name = update_ds(user_demand,ideas,iter,roomtype)
            update_infinigen(action, iter, json_name)

        elif action == "finish":
            update_infinigen(action, iter, json_name)
            break

        else:
            raise ValueError(f"Action is wrong: {action}")
    


    grades = eval_scene(iter,user_demand)

    iter += 1
    # update_infinigen(action,iter,json_name)
    # solve_action = choose_solve()
    # update_infinigen("solve_large",iter)
    # update_infinigen("solve_large_and_medium",iter)
    # update_infinigen("solve_small",iter)


action = "finalize_scene"
update_infinigen(action, iter, json_name="empty")
