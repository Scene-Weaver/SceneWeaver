import bpy
import trimesh
import json
# from infinigen.core import tags as t
from infinigen.core.constraints.evaluator.node_impl.trimesh_geometry import any_touching
from infinigen.core.constraints.constraint_language import util as iu

def eval_metric(
    state,
    iter
):
    Nobj, OOB, BBL = eval_physics_score(state)
    # real, func, complet = eval_general_score(state)
    # GPT_sim, CLIP_sim = eval_align_score(state)

    results = {
        "Nobj":Nobj, 
        "OOB":OOB,
        "BBL":BBL,
        # "real":real,
        # "func":func,
        # "complet":complet,
        # "GPT_sim":GPT_sim,
        # "CLIP_sim":CLIP_sim,
    }

    with open(f"record_files/metric_{iter}.json", "w") as file:
        json.dump(results, file,indent=4)
    return





# def eval_align_score(state):
#     GPT_sim = 0
#     CLIP_sim = 0
#     return GPT_sim, CLIP_sim



def eval_physics_score(state):
    scene = state.trimesh_scene
    collision_objs = []
    for name, info in state.objs.items():
        if name.startswith("window") or name == "newroom_0-0" or name == "entrance":
            continue
        else:
            collision_objs.append(state.objs[name].populate_obj)  #mesh
            # collision_objs.append(state.objs[name].obj.name)  #bbox
    # collision_objs = [
    #     os.obj.name
    #     for k, os in state.objs.items()
    #     if t.Semantics.NoCollision not in os.tags
    # ]
    Nobj = len(collision_objs)
    print("Nobj: ",Nobj)
    
    OOB_objs = []
    room_obj = state.objs["newroom_0-0"].obj
    normal_b = [0,0,1]
    origin_b = [0,0,0]
    b_trimesh = iu.meshes_from_names(scene, room_obj.name)[0]
    projected_b = trimesh.path.polygons.projected(b_trimesh, normal_b, origin_b)
    for name in collision_objs:
        # target_obj = bpy.data.objects.get(name)
        a_trimesh = iu.meshes_from_names(scene, name)[0]
        # try:
        #     projected_a = trimesh.path.polygons.projected(a_trimesh, normal_b, origin_b)
        # except:
        #     projected_a = trimesh.path.polygons.projected(a_trimesh.convex_hull, normal_b, origin_b)
        projected_a = trimesh.path.polygons.projected(a_trimesh.convex_hull, normal_b, origin_b)
        res = projected_a.within(projected_b.buffer(1e-2))
        if not res:
            OOB_objs.append(name)
    OOB = len(OOB_objs)
    print("OOB: ",OOB)
    
    touch = any_touching(
        scene,
        collision_objs,
        collision_objs,
        bvh_cache=state.bvh_cache
    )
    collide_pairs = [[name1,name2] for name1,name2 in touch.names if name1!=name2]
    # collide_pairs = [[max(name1,name2),min(name1,name2)] for name1,name2 in touch.names if name1!=name2]
    # collide_pairs = set(collide_pairs)
    BBL = len(collide_pairs)
    print("BBL: ",BBL)

    return Nobj, OOB, BBL


def eval_general_score(image_path_1,layout,image_path_2=None):
    # real = 0
    # func = 0
    # complet = 0
    
    # return real, func, complet

    import base64
    import requests
    import json
    import numpy as np
    import re
    import argparse


    # TODO : OpenAI API Key
    api_key = "YOUR_API_KEY"

    # TODO : Path to your image
    image_path_1 = "FIRST_IMAGE_PATH.png"
    image_path_2 = "SECOND_IMAGE_PATH.png"

    # TODO : User preference Text
    user_preference = "USER_PREFERNCE_TEXT"

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    example_json ="""
    {
    "realism_and_3d_geometric_consistency": {
        "grade": 8,
        "comment": "The renders appear to have appropriate 3D geometry and lighting that is fairly consistent with real-world expectations. The proportions and perspective look realistic."
    },
    "functionality_and_activity_based_alignment": {
        "grade": 7,
        "comment": "The room includes a workspace, sleeping area, and living area as per the user preference. The L-shaped couch facing the bed partially meets the requirement for watching TV comfortably. However, there does not appear to be a TV depicted in the render, so it's not entirely clear if the functionality for TV watching is fully supported."
    },
    "layout_and_furniture": {
        "grade": 7,
        "comment": "The room has a bed that’s not centered and with space at the foot, and a large desk with a chair. However, it's unclear if the height of the bed meets the user's preference, and the layout does not clearly show the full-length mirror in relation to the wardrobe, so its placement in accordance to user preferences is uncertain."
    },
    "completion_and_richness_of_detail": {
        "grade": 9,
        "comment": "The render includes detailed elements such as books on the desk, a rug under the coffee table, and small decorative items on the shelves. These touches add a sense of realism and completeness to the room, making it feel lived-in and thoughtfully designed."
    }
    """

    # Getting the base64 string
    base64_image_1 = encode_image(image_path_1)
    # base64_image_2 = encode_image(image_path_2)


    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"""
            Give a grade from 1 to 10 or unknown to the following room renders and layout based on how well they correspond together to the user preference (in triple backquotes) in the following aspects: 
            - Realism and 3D Geometric Consistency
            - Functionality and Activity-based Alignment
            - Layout and furniture     
            - Completion and richness of detail  
            User Preference:
            ```{user_preference}```
            Room layout:
            ```{layout}```
            Return the results in the following JSON format:
            ```json
            {example_json}
            ```
            """
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image_1}"
            }
            },
            # {
            # "type": "image_url",
            # "image_url": {
            #     "url" : f"data:image/jpeg;base64,{base64_image_2}"
            # }
            # }
        ]
        }
    ],
    "max_tokens": 1024
    }
    grades = {
    "realism_and_3d_geometric_consistency": [],
    "functionality_and_activity_based_alignment": [],
    "layout_and_furniture": [],
    # "color_scheme_and_material_choices": [],
    "completion_and_richness_of_detail": []
    }
    for _ in range(3):
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        grading_str = response.json()["choices"][0]["message"]["content"]
        print(grading_str)
        print("-" * 50)
        pattern = r'```json(.*?)```'
        matches = re.findall(pattern, grading_str, re.DOTALL)
        json_content = matches[0].strip() if matches else None
        if json_content is None:
            grading = json.loads(grading_str)
        else:
            grading = json.loads(json_content)
        for key in grades:
            grades[key].append(grading[key]["grade"])
    #Save the mean and std of the grades
    for key in grades:
        grades[key] = {"mean": round(sum(grades[key])/len(grades[key]), 2), "std": round(np.std(grades[key]), 2)}
    #Save the grades
    with open(f"{'_'.join(image_path_1.split('_')[:-1])}_grades.json", "w") as f:
        json.dump(grades, f)     

    return grades
