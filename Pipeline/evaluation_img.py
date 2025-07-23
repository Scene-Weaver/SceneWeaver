import json
import re
import time

import numpy as np
from gpt import GPT4


def eval_scene(iter, user_demand):
    grades, grading = eval_general_score(iter, user_demand)

    with open(
        f"~/workspace/SceneWeaver/record_files/metric_{iter}.json", "r"
    ) as f:
        results = json.load(f)

    metric = dict()
    metric["GPT score (0-10, higher is better)"] = grading
    metric["Physics score"] = {
        "object number (higher is better)": results["Nobj"],
        "object not inside the room (less is better)": results["OOB"],
        "object has collision (less is better)": results["BBL"],
    }

    json_name = f"~/workspace/SceneWeaver/Pipeline/record/metric_{iter}.json"
    with open(json_name, "w") as f:
        json.dump(metric, f, indent=4)

    return metric


def eval_general_score(iter, user_demand):
    # basedir = "/mnt/fillipo/yandan/scenesage/record_scene/manus/well_designed_bedroom_with__realistic_layout/record_scene/"
    basedir = "~/workspace/SceneWeaver/record_scene/"
    image_path_1 = f"{basedir}/render_{iter}.jpg"

    # gpt = GPT4(version="4.1",region="eastus2")
    gpt = GPT4(version="4o")

    example_json = """
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
    Positive: Objects are well placed. Room is clean and tidy. Relation between objects are reasonable, such as chair face to the desk. Layout matches the user's preference. Orientation of each object is correct.
    Negative: Room is messy. Objects are placed in wrong places or placed randomly. Floating objects. Some large objects do not stand close to the wall when they are supposed to, such as large shelf and sofa. Orientation of some object is incorrect.
4. Completion: How complete the room is, considering both large furniture and smaller objects like decor, accessories, and everyday items. 
    Positive: Do not need more objects. A complete room correspond to user's preference.
    Negative: Room is empty. Too many blank areas. Seems unfinished.

Use the following user preferences as reference (enclosed in triple backticks):
User Preference:
```{user_demand}```

Note different category name can represent the same category, such as ChairFactory, armchair and chair can represent chair simultaneously.
Count objects carefully! Do not miss any details.

Return the results in the following JSON format, the "comment" should be short:
{example_json}

For the image:
Each object is marked with a 3D bounding box and its category label. You must count the object carefully with the given image.

You are working in a 3D scene environment with the following conventions:

- Right-handed coordinate system.
- The X-Y plane is the floor.
- X axis (red) points right, Y axis (green) points top, Z axis (blue) points up.
- For the location [x,y,z], x,y means the location of object's center in x- and y-axis, z means the location of the object's bottom in z-axis.
- All asset local origins are centered in X-Y and at the bottom in Z.
- By default, assets face the +X direction.
- A rotation of [0, 0, 1.57] in Euler angles will turn the object to face +Y.
- All bounding boxes are aligned with the local frame and marked in blue with category labels.
- The front direction of objects are marked with yellow arrow.
- Coordinates in the image are marked from [0, 0] at bottom-left of the room.

"""

    prompt_payload = gpt.get_payload_eval(
        prompting_text_user=prompting_text_user, render_path=image_path_1
    )

    grades = {"realism": [], "functionality": [], "layout": [], "completion": []}
    for _ in range(1):
        try:
            grading_str = gpt(payload=prompt_payload, verbose=True)
        except:
            time.sleep(30)
            grading_str = gpt(payload=prompt_payload, verbose=True)
        print(grading_str)
        print("-" * 50)
        pattern = r"```json(.*?)```"
        matches = re.findall(pattern, grading_str, re.DOTALL)
        json_content = matches[0].strip() if matches else None
        if json_content is None:
            grading = json.loads(grading_str)
        else:
            grading = json.loads(json_content)
        for key in grades:
            grades[key].append(grading[key]["grade"])

    with open(f"metric_{iter}.json", "w") as f:
        json.dump(grading, f, indent=4)
    # Save the mean and std of the grades
    for key in grades:
        grades[key] = {
            "mean": round(sum(grades[key]) / len(grades[key]), 2),
            "std": round(np.std(grades[key]), 2),
        }
    # Save the grades
    # with open(f"metric_{iter}.json", "w") as f:
    #     json.dump(grades, f, indent=4)

    return grades, grading


if __name__ == "__main__":
   
    eval_scene(
        11,
        "Design me a classroom.",
    )
