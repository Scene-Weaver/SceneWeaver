import json
import os

from gpt import GPT4

from app.prompt.acdc_cand import system_prompt, user_prompt
from app.tool.base import BaseTool
from app.tool.update_infinigen import update_infinigen
from app.utils import extract_json

DESCRIPTION = """
Using image generation and 3D reconstruction to add additional objects into the current scene.

Use Case 1: Add **a group of** small objects on the top of an empty and large furniture, such as a table, cabinet, and desk when there is nothing on its top. 

You **MUST** not:
1.Do not add objects where there is no available space.
2.Do not add objects where there already exists other small objects.
3.Do not add small objects on any tall furniture, such as wardrob.
4.Do not add small objects on small supporting surface, such as nightstand.
5.Do not add small objects on concave furniture, such as sofa and shelf.

Strengths: Real. Excellent for adding a group of objects with inter-relations on the top of a large furniture.(e.g., enriching a tabletop), such as adding (laptop,mouse,keyboard) set on the desk and (plate,spoon,food) set on the dining table. Accurate in rotation. 
Weaknesses: Very slow. Can not add objects on the wall, ground, or ceiling. Can not add objectsinside a container, such as objects in the shelf. Can not add objects when there is already something on the top.

"""


class AddAcdcExecute(BaseTool):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "add_acdc"
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
        # 1 generate prompt for sd + 2 use sd to generate image + 3 use acdc to reconstruct 3D scene
        user_demand = os.getenv("UserDemand")
        iter = int(os.getenv("iter"))
        roomtype = os.getenv("roomtype")
        action = self.name
        try:
            # 1 generate prompt for sd
            steps = gen_ACDC_cand(user_demand, ideas, roomtype, iter)

            inplace = False
            acdc_record = dict()
            for obj_id, info in steps.items():
                sd_prompt = info["prompt for SD"]
                if sd_prompt not in acdc_record:
                    update_infinigen(
                        "export_supporter", iter, json_name="", description=obj_id
                    )
                    cnt = 0
                    while True and cnt < 5:
                        cnt += 1
                        print(sd_prompt)
                        # 2 use sd to generate image
                        img_filename = gen_img_SD(
                            sd_prompt, obj_id, info["obj_size"]
                        )  # execute until satisfy the requirement

                        # 3 use acdc to reconstruct 3D scene
                        _ = acdc(img_filename, obj_id, info["obj category"])

                        with open(
                            "~/workspace/digital-cousins/args.json", "r"
                        ) as f:
                            j = json.load(f)
                            if j["success"]:
                                save_dir = os.getenv("save_dir")
                                newid = obj_id.replace(" ", "_")
                                foldername_old = f"{save_dir}/pipeline/acdc_output/step_3_output/scene_0/"
                                foldername_new = f"{save_dir}/pipeline/{newid}"
                                os.system(f"cp -r {foldername_old} {foldername_new}")
                                json_name = f"{foldername_new}/scene_0_info.json"
                                acdc_record[sd_prompt] = json_name
                                break
                    assert j["success"]
                else:
                    json_name = acdc_record[sd_prompt]

                update_infinigen(
                    action,
                    iter,
                    json_name,
                    description=obj_id,
                    inplace=inplace,
                    ideas=ideas,
                )
                inplace = True

            return "Successfully add objects with ACDC."
        except Exception:
            return "Error adding objects with ACDC"


def acdc(img_filename, obj_id, category):
    # objtype = obj_id.split("_")[1:]
    # objtype = "_".join(objtype)
    j = {
        "obj_id": obj_id,
        "objtype": category.lower(),
        "img_filename": img_filename,
        "success": False,
        "error": "Unknown",
    }
    with open("~/workspace/Tabletop-Digital-Cousins/args.json", "w") as f:
        json.dump(j, f, indent=4)

    import subprocess

    cmd = """
    source /home/yandan/anaconda3/etc/profile.d/conda.sh
    conda deactivate
    cd ~/workspace/Tabletop-Digital-Cousins
    conda activate acdc2
    python digital_cousins/pipeline/acdc_pipeline.py --gpt_api_key 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 > ~/workspace/SceneWeaver/Pipeline/run.log 2>&1
    """
    subprocess.run(["bash", "-c", cmd])
    # os.system("bash -i ~/workspace/digital-cousins/run.sh")
    save_dir = os.getenv("save_dir")
    json_name = (
        f"{save_dir}/pipeline/acdc_output/step_3_output/scene_0/scene_0_info.json"
    )

    return json_name


def gen_img_SD(SD_prompt, obj_id, obj_size):
    # objtype = obj_id.split("_")[1:]
    # objtype = "_".join(objtype)
    # SD_prompt = gen_SD_prompt(prompt,objtype,obj_size)
    save_dir = os.getenv("save_dir")
    img_filename = f"{save_dir}/pipeline/SD_img.jpg"
    j = {"prompt": SD_prompt, "img_savedir": img_filename}
    with open("~/workspace/sd3.5/prompt.json", "w") as f:
        json.dump(j, f, indent=4)

    basedir = "~/workspace/sd3.5"
    os.system(f"bash {basedir}/run.sh")

    return img_filename


def gen_ACDC_cand(user_demand, ideas, roomtype, iter):
    save_dir = os.getenv("save_dir")
    with open(f"{save_dir}/record_scene/layout_{iter-1}.json", "r") as f:
        layout = json.load(f)
    layout = layout["objects"]

    # convert size
    for key in layout.keys():
        size = layout[key]["size"]
        size_new = [size[1], size[0], size[2]]
        layout[key]["size"] = size_new

    gpt = GPT4(version="4.1")

    user_prompt_1 = user_prompt.format(
        user_demand=user_demand, ideas=ideas, roomtype=roomtype, scene_layout=layout
    )

    prompt_payload = gpt.get_payload(system_prompt, user_prompt_1)

    gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    print(gpt_text_response)
    results = extract_json(gpt_text_response)

    with open(f"{save_dir}/pipeline/acdc_candidates_{iter}.json", "w") as f:
        json.dump(results, f, indent=4)

    return results
