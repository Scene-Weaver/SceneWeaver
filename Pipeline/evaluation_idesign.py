import json
import os
import re
import time

import numpy as np
from app.utils import dict2str
from gpt import GPT4


def diff_objects(iter):
    save_dir = os.getenv("save_dir")

    with open(f"{save_dir}/record_scene/layout_{iter}.json", "r") as f:
        layout = json.load(f)
        layout = layout["objects"]
        objs_now = layout.keys()

    if iter == 0:
        return {"newly added objects": list(objs_now), "removed objects": []}

    with open(f"{save_dir}/record_scene/layout_{iter-1}.json", "r") as f:
        layout = json.load(f)
        layout = layout["objects"]
        objs_past = layout.keys()

    new_objs = list(set(objs_now) - set(objs_past))
    remove_obs = list(set(objs_past) - set(objs_now))
    return {"newly added objects": new_objs, "removed objects": remove_obs}


def statistic_traj(iter):
    save_dir = os.getenv("save_dir")
    trajs = dict()
    for i in range(iter + 1):
        traj = dict()
        traj["iter"] = i
        # with open(f"{save_dir}/args/args_{i}.json","r") as f:
        #     j = json.load(f)
        #     traj["action"] = j["action"]
        #     traj["ideas"] = j["ideas"]
        with open(f"{save_dir}/pipeline/metric_{i}.json", "r") as f:
            j = json.load(f)
            traj["results"] = dict()
            traj["results"]["GPT score (0-10, higher is better)"] = j[
                "GPT score (0-10, higher is better)"
            ]
            traj["results"]["Object Difference"] = j["Object Difference"]

        with open(f"{save_dir}/record_files/metric_{iter}.json", "r") as f:
            j = json.load(f)
            traj["results"]["Physics score"] = {
                "object number": j["Nobj"],
                "object not inside the room": j["OOB Objects"],
                "object has collision": j["BBL objects"],
            }

        trajs[i] = traj

    with open(f"{save_dir}/pipeline/trajs_{iter}.json", "w") as f:
        json.dump(trajs, f, indent=4)

    return trajs


def eval_scene(iter, user_demand, verbo=True):
    grades, grading = eval_general_score(iter, user_demand, verbo)
    obj_diff = diff_objects(iter)

    save_dir = os.getenv("save_dir")
    with open(f"{save_dir}/record_files/metric_{iter}.json", "r") as f:
        results = json.load(f)

    metric = dict()
    metric["Object Difference"] = obj_diff
    metric["GPT score (0-10, higher is better)"] = grading
    # metric["Physics score"] = {
    #     "object number (higher is better)": results["Nobj"],
    #     "object not inside the room (lower is better)": results["OOB"],
    #     "object has collision (lower is better)": results["BBL"],
    #     # "object not inside the room (lower is better)": results["OOB Objects"],
    #     # "object has collision (lower is better)": results["BBL objects"],
    # }

    save_dir = os.getenv("save_dir")
    json_name = f"{save_dir}/pipeline/metric_{iter}.json"
    with open(json_name, "w") as f:
        json.dump(metric, f, indent=4)

    action_trajs = statistic_traj(iter)

    return metric


def eval_general_score(iter, user_demand, verbo=True):
    save_dir = os.getenv("save_dir")
    if not os.path.exists(f"{save_dir}/pipeline/"):
        os.mkdir(f"{save_dir}/pipeline/")
    # basedir = "/mnt/fillipo/yandan/scenesage/record_scene/bedroom/record_scene"
    image_path_1 = f"{save_dir}/record_scene/render_{iter}_marked.jpg"
    with open(f"{save_dir}/record_scene/layout_{iter}.json", "r") as f:
        layout = json.load(f)
        layout = layout["objects"]
        layout = dict2str(layout)

    # gpt = GPT4(version="4.1",region="eastus2")
    gpt = GPT4(version="4.1")

    example_json = """
{
  "realism": {
    "grade": your grade as int,
    "comment": "Your comment and suggestion."
  },
  "functionality": {
    "grade": your grade as int,
    "comment": "Your comment and suggestion."
  },
  "layout": {
    "grade": your grade as int,
    "comment": "Your comment and suggestion."
  },
  "completion": {
    "grade": your grade as int,
    "comment": "Your comment and suggestion."
  }
}
    """

    prompting_text_user = f"""
You are given a top-down room render image and the corresponding layout of each object. 
Your task is to evaluate how well they align with the user’s preferences (provided in triple backticks) across the four criteria listed below.
For each criterion, assign a score from 0 to 10, and provide a brief justification for your rating.

Scoring must be strict. If any critical issue is found (such as missing key objects, obvious layout errors, or unrealistic elements), the score should be significantly lowered, even if other aspects are fine.

**Score Guidelines**:
- Score 10: Fully meets or exceeds expectations; no major improvements needed.
- Score 5: Partially meets expectations; some obvious flaws exist that limit usefulness or quality.
- Score 0: Completely fails to meet expectations; the aspect is absent, wrong, or contradicts user needs.

**Evaluation Criteria**:

1. **Realism**: How realistic the room appears. *Ignore texture, lighting, and doors.*
    - **Good (8-10)**: The layout (position, rotation, and size) is believable, and common daily objects make the room feel lived-in. Rich of daily furniture and objects.
    - **Bad (0-3)**: Unusual objects or strange placements make the room unrealistic.
    - **Note**: If object types or combinations defy real-world logic (e.g., bathtubs in bedrooms), score should be below 5.

2. **Functionality**: How well the room supports the intended activities (e.g., sleeping, working).
    - **Good (8-10)**: Contains the necessary furniture and setup for the specified function.
    - **Bad (0-3)**: Missing key objects or contains mismatched furniture (e.g., no bed in a bedroom).
    - **Note**: Even one missing critical item should lower the score below 6.

3. **Layout**: Whether the furniture is arranged logically in good pose and aligns with the user’s preferences.
    - **Good (8-10)**: Each objects is in **reasonable size**, neatly placed, objects of the same category are well aglined, relationships are reasonable (e.g., chairs face desks), sufficient space exists for walking, and **orientations must** be correct. 
    - **Bad (0-3)**: Floating objects, crowded floor, **abnormal size**, objects with collision, incorrect **orientation**, or large items placed oddly (e.g., sofa not against the wall). Large empty space. Blocker in front of furniture.
    - **Note**: If the room has layout issues that affect use, it should not score above 5.

4. **Completion**: How complete and finished the room feels.
    - **Good (8-10)**: All necessary large and small items are present. Has rich details. Each shelf is full of objects (>5) inside. Each supporter (e.g. table, desk, and shelf) has small objects on it. Empty area is less than 50%. The room feels done.
    - **Bad (0-3)**: Room is sparse or empty, lacks decor or key elements.
    - **Note**: If more than 30% of the room is blank or lacks detail, score under 5.


Use the following user preferences as reference (enclosed in triple backticks):
User Preference:
```{user_demand}```

Room layout:
{layout}

The Layout include each object's X-Y-Z Position, rotation, size (length, width, height) in meter, as well as relation info with parents.
Each key in layout is the name for each object, consisting of a random number and the category name, such as "3142143_table". 
Note different category name can represent the same category, such as ChairFactory, armchair and chair can represent chair simultaneously.
Count objects carefully! Do not miss any details. 
Pay more attention to the orientation of each objects.
Pay more attention to the size of each "ontop" objects.

Return the results in the following JSON format, the "comment" should be short:
{example_json}

For the image:
Each object is marked with a 3D bounding box and its category label. You must count the object carefully with the given image and layout.

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
        if verbo:
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

    with open(f"{save_dir}/pipeline/grade_iter_{iter}.json", "w") as f:
        json.dump(grading, f, indent=4)
    # Save the mean and std of the grades
    for key in grades:
        grades[key] = {
            "mean": round(sum(grades[key]) / len(grades[key]), 2),
            "std": round(np.std(grades[key]), 2),
        }
    # Save the grades
    with open(f"{save_dir}/pipeline/eval_iter_{iter}.json", "w") as f:
        json.dump(grades, f, indent=4)

    return grades, grading


if __name__ == "__main__":
    method = "idesign"  # "classroom","grameroom", \
    # roomtype = "bedroom" #idesign bathroom [7.4, 7.2, 7.4, 5.4, 9.6, 0.0, 0.0] #"childrenroom"
    for roomtype in ["bookstore"]:
        for i in range(5):
            try:
                save_dir = f"/mnt/fillipo/yandan/scenesage/record_scene/{method}/{roomtype}/scene_{i}"
                os.environ["save_dir"] = save_dir
                print(f"evaluating {method} {roomtype}_{i}")
                if not os.path.exists(f"{save_dir}/record_files/metric_0.json"):
                    continue
                if os.path.exists(f"{save_dir}/pipeline/trajs_0.json"):
                    continue
                metric = eval_scene(0, f"Design me a {roomtype}.", verbo=False)
            except:
                continue
        score = {
            "gpt_real": [],
            "gpt_function": [],
            "gpt_layout": [],
            "gpt_complet": [],
            "NObj": [],
            "BBO": [],
            "BBL": [],
        }

        # for i in [10,12,14,15,16,17,19]:
        try:
            for i in range(5):
                save_dir = f"/mnt/fillipo/yandan/scenesage/record_scene/{method}/{roomtype}/scene_{i}"
                if not os.path.exists(f"{save_dir}/pipeline/trajs_0.json"):
                    continue
                with open(f"{save_dir}/pipeline/trajs_0.json", "r") as f:
                    j = json.load(f)

                gptscore = j["0"]["results"]["GPT score (0-10, higher is better)"]
                score["gpt_real"].append(gptscore["realism"]["grade"])
                score["gpt_function"].append(gptscore["functionality"]["grade"])
                score["gpt_layout"].append(gptscore["layout"]["grade"])
                score["gpt_complet"].append(gptscore["completion"]["grade"])

                physcore = j["0"]["results"]["Physics score"]
                score["NObj"].append(physcore["object number"])
                score["BBO"].append(len(physcore["object not inside the room"]))
                score["BBL"].append(len(physcore["object has collision"]))

            score_mean = dict()
            for key, value in score.items():
                score_mean[key] = round(np.mean(value), 3)

            print(method, roomtype, list(score_mean.values()))
        except:
            continue
