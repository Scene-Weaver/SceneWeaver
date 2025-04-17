import get_action_prompt as prompts
from gpt import GPT4 
from utils import extract_json, dict2str, lst2str
import json


system_prompt = prompts.system_prompt
methods_prompt = prompts.methods_prompt
idea_example = prompts.idea_example

def load_previous_guide(iter):
    previous_guide = []
    for i in range(iter):
        with open(f"/home/yandan/workspace/infinigen/Pipeline/record/metric_{i}.json","r") as f:
            m = json.load(f)
        with open(f"record/get_action_iter_{i}.json","r") as f:
            j = json.load(f)
        info = {"iter": j["iter"],
                # "Thoughts": j["Thoughts"],
                # "Recommendation": j["Recommendation"],
                "Method number": j["Method number"],
                "Ideas": j["Ideas"],
                "Evaluate Score": m
                }
        roomtype = j["RoomType"]
        previous_guide.append(dict2str(info))

    return previous_guide, roomtype


def get_action0(user_demand = "Classroom",iter = 0):

    previous_guide = "None"

    sceneinfo_prompt =  prompts.sceneinfo_prompt.format(scene_layout="None")

    
    feedback_reflections_system_payload = prompts.feedback_reflections_prompt_system.format(system_prompt=system_prompt,
                                                                                                    methods_prompt=methods_prompt)
    
    feedback_reflections_user_payload = prompts.feedback_reflections_prompt_user.format(iter=iter,
                                                                                        user_demand=user_demand,
                                                                                        roomtype="To be defined by 1-2 words.",
                                                                                        previous_guide=previous_guide,
                                                                                        sceneinfo_prompt=sceneinfo_prompt,
                                                                                        idea_example=idea_example
                                                                                        )
    gpt = GPT4(version="4o")

    prompt_payload = gpt.get_payload_scene_image(feedback_reflections_system_payload, feedback_reflections_user_payload,render_path=None )
    gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    print(gpt_text_response)
    gpt_dict_response = extract_json(gpt_text_response)

    with open(f"record/get_action_iter_{iter}.json","w") as f:
        gpt_dict_response["user_demand"] = user_demand
        gpt_dict_response["iter"] = iter
        gpt_dict_response["system_prompt"] = system_prompt
        gpt_dict_response["previous_guide"] = previous_guide
        gpt_dict_response["sceneinfo_prompt"] = sceneinfo_prompt
        gpt_dict_response["idea_example"] = idea_example
        gpt_dict_response["methods_prompt"] = methods_prompt
        json.dump(gpt_dict_response,f,indent=4)

   
    method_number = gpt_dict_response["Method number"]
    init_methods = [None,"init_metascene","init_physcene",None,"init_gpt"]
    
    action = init_methods[method_number]
    ideas = gpt_dict_response["Ideas"]
    roomtype = gpt_dict_response["RoomType"]

    return action,ideas,roomtype



def get_action1(user_demand = "",iter = 1):

    previous_guide,roomtype = load_previous_guide(iter)
    previous_guide = lst2str(previous_guide)

    render_path = f"/home/yandan/workspace/infinigen/record_scene/render_{iter-1}.jpg"
    with open(f"/home/yandan/workspace/infinigen/record_scene/layout_{iter-1}.json", "r") as f:
        layout = json.load(f)
    sceneinfo_prompt =  prompts.sceneinfo_prompt.format(scene_layout=dict2str(layout["objects"]))

    idea_example = prompts.idea_example
    feedback_reflections_system_payload = prompts.feedback_reflections_prompt_system.format(system_prompt=system_prompt,
                                                                                                    methods_prompt=methods_prompt)
    feedback_reflections_user_payload = prompts.feedback_reflections_prompt_user.format(iter=iter,
                                                                                        user_demand=user_demand,
                                                                                        roomtype=roomtype,
                                                                                        previous_guide=previous_guide,
                                                                                        sceneinfo_prompt=sceneinfo_prompt,
                                                                                        idea_example=idea_example
                                                                                        )
    # gpt = GPT4(version='4.5',region="eastus2")

    gpt = GPT4(version='4o')
    
    prompt_payload = gpt.get_payload_scene_image(feedback_reflections_system_payload, feedback_reflections_user_payload,render_path=render_path)
    gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    print(gpt_text_response)
    gpt_dict_response = extract_json(gpt_text_response)

    with open(f"record/get_action_iter_{iter}.json","w") as f:
        gpt_dict_response["user_demand"] = user_demand
        gpt_dict_response["iter"] = iter
        gpt_dict_response["system_prompt"] = system_prompt
        gpt_dict_response["previous_guide"] = previous_guide
        gpt_dict_response["sceneinfo_prompt"] = sceneinfo_prompt
        gpt_dict_response["idea_example"] = idea_example
        gpt_dict_response["methods_prompt"] = methods_prompt
        json.dump(gpt_dict_response,f,indent=4)

    method_number = gpt_dict_response["Method number"]
    init_methods = ["finish",None,None,"add_acdc","add_gpt","update"]


    action = init_methods[method_number]
    ideas = gpt_dict_response["Ideas"]
    roomtype = gpt_dict_response["RoomType"]

    return action,ideas,roomtype
