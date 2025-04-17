import json
import base64
import requests
import numpy as np
import re
from gpt import GPT4 
from utils import dict2str
import time

def eval_scene(iter,user_demand):
    grades, grading = eval_general_score(iter, user_demand)

    with open(f"/home/yandan/workspace/infinigen/record_files/metric_{iter}.json","r") as f:
        results = json.load(f)

    metric = dict()
    metric["GPT score (0-10, higher is better)"] = grading
    metric["Physics score"] = {
        "object number (higher is better)":results["Nobj"] ,
        "object not inside the room (lower is better)":results["OOB"],
        "object has collision (lower is better)":results["BBL"] 
    }
    

    json_name = f"/home/yandan/workspace/infinigen/Pipeline/record/metric_{iter}.json"
    with open(json_name, "w") as f:
        json.dump(metric, f, indent=4)

    return metric


def eval_general_score(iter,user_demand):
    basedir = "/home/yandan/workspace/infinigen/record_scene"
    # basedir = "/mnt/fillipo/yandan/scenesage/record_scene/bedroom/record_scene"
    image_path_1 = f"{basedir}/render_{iter}.jpg"
    with open(f"{basedir}/layout_{iter}.json", "r") as f:
        layout = json.load(f)
        layout = layout["objects"]
        layout = dict2str(layout)

    gpt = GPT4(version='4o')

    example_json ="""
{
  "realism": {
    "grade": your grade as int,
    "comment": "Your comment."
  },
  "functionality": {
    "grade": your grade as int,
    "comment": "Your comment."
  },
  "layout": {
    "grade": your grade as int,
    "comment":"Your comment."
  },
  "completion": {
    "grade": your grade as int,
    "comment": "Your comment."
  }
}
    """

    prompting_text_user = f"""
You are given a top-down room render image and the corresponding layout of each object. 
Your task is to evaluate how well they align with the user’s preferences (provided in triple backticks) across the four criteria listed below.
For each criterion, assign a score from 0 to 10, and provide a brief justification for your rating. 

Scoring Notes:
Score of 10: Fully meets or exceeds expectations for this aspect; no major improvements needed.
Score of 0: Completely fails to meet expectations; this aspect is either absent or directly contradicts the preferences.
    
Evaluation Criteria:
1. Realism: How realistic the room appears according to the user's prefernce. Do not pay attention to the texture, lighting, and door.
    Positive: Room has real layout. Multiple daily objects contribute to a lived-in, believable feel.
    Negative: Room has strange objects and layout, which is far from a real room.
2. Functionality: How well the room supports the activities specified in the user’s preferences (e.g., sleeping, working, relaxing, watching TV).
    Positive: The room is designed for function of user's preference. Objects in the room is designed for the related activity.
    Negative: The room does not satisfy the function and activity. The room type mismatch the requirement. In lack of the key object.
3. Layout: How logically and efficiently the furniture is arranged, and whether the layout matches user preferences for spacing, orientation, or relations between objects.
    Positive: Objects are well placed. Room is clean and tidy. Relation between objects are reasonable, such as chair face to the desk. Layout matches the user's preference.
    Negative: Room is messy. Objects are placed in wrong places or placed randomly. Floating objects. Some large objects do not stand close to the wall when they are supposed to, such as large shelf and sofa.
4. Completion: How complete the room is, considering both large furniture and smaller objects like decor, accessories, and everyday items. 
    Positive: Do not need more objects. A complete room correspond to user's preference.
    Negative: Room is empty. Too many blank areas. Seems unfinished.

Use the following user preferences as reference (enclosed in triple backticks):
User Preference:
```{user_demand}```

Room layout:
{layout}

The Layout include each object's X-Y-Z Position, Z rotation, size (x_dim, y_dim, z_dim), as well as relation info with parents.
Each key in layout is the name for each object, consisting of a random number and the category name, such as "3142143_table". 
Note different category name can represent the same category, such as ChairFactory, armchair and chair can represent chair simultaneously.
Count objects carefully! Do not miss any details.

Return the results in the following JSON format:
{example_json}

For the image:
Each object is marked with a 3D bounding box and its category label. You must count the object carefully with the given image and layout.
**3D Convention:**
- Right-handed coordinate system.
- The X-Y plane is the floor; the Z axis points up. The origin is at a corner (the left-top corner of the rendered image), defining the global frame.
- Asset front faces point along the positive X axis. The Z axis points up. The local origin is centered in X-Y and at the bottom in Z. 
A 90-degree Z rotation means that the object will face the positive Y axis. The bounding box aligns with the assets local frame.

    """


    prompt_payload = gpt.get_payload_eval(prompting_text_user=prompting_text_user)

    grades = {
        "realism": [],
        "functionality": [],
        "layout": [],
        "completion": []
    }
    for _ in range(1):
        try:
            grading_str = gpt(payload=prompt_payload, verbose=True)
        except:
            time.sleep(30)
            grading_str = gpt(payload=prompt_payload, verbose=True)
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

    with open(f"record/grade_iter_{iter}.json","w") as f:
        json.dump(grading, f, indent=4)  
    #Save the mean and std of the grades
    for key in grades:
        grades[key] = {"mean": round(sum(grades[key])/len(grades[key]), 2), "std": round(np.std(grades[key]), 2)}
    #Save the grades
    with open(f"record/eval_iter_{iter}.json","w") as f:
        json.dump(grades, f, indent=4)     

    return grades,grading

if __name__ == "__main__":
    # grades = {
    #     "realism": {"mean":[],"std":[]},
    #     "functionality":  {"mean":[],"std":[]},
    #     "layout": {"mean":[],"std":[]},
    #     "completion": {"mean":[],"std":[]},
    #     "OOB":[],
    #     "BBL":[],
    #     "Nobj":[]
    # }
    # for i in [9,8,7,915]:
    #     print(f"*** iter {i} ***")
    #     eval_scene(i,"Design me a bedroom.")
    #     # eval_scene(i,"A game room for a 6-year-old boy.")

    #     with open(f"record/eval_iter_{i}.json","r") as f:
    #         j = json.load(f)  
    #     for key,value in j.items():  
    #         grades[key]["mean"].append(value["mean"])
    #         grades[key]["std"].append(value["std"])

    #     with open(f"/home/yandan/workspace/infinigen/record_files/metric_{i}.json","r") as f:
    #         j = json.load(f)  
    #     for key,value in j.items():  
    #         grades[key].append(value)
            
    
    # with open(f"record/eval_iter_0_{i}.json","w") as f:
    #     json.dump(grades, f) 
    eval_scene(3, "You must design a scene iteratively using the tools I designed, it must have one large table with eight chairs placing properly next to the table with appropriate size and scale. You can choose to modify the scene by adding and eliminating objects. It should have a large table with comfortable seating for the family and guests.")