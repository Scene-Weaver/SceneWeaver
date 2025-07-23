import json
import os
import sys

sys.path.append("~/workspace/SceneWeaver/Pipeline")
from gpt import GPT4
from metascene_frontview import calculate_object_widths, filter_side_img


def get_obj_frontview(objav_path, category, verbose=False):
    outpath = f"{objav_path}/metadata.json"
    if os.path.exists(outpath):
        return

    gpt = GPT4(version="4.1")

    candidates_fpaths = []
    out_dict = dict()

    if not os.path.exists(f"{objav_path}"):
        print("missing resource", objav_path)
        return False

    if verbose:
        print("processing ", objav_path)

    inrenderdir = objav_path
    candidates_fpaths = []
    for file in os.listdir(inrenderdir):
        candidates_fpaths.append(f"{inrenderdir}/{file}")
    candidates_fpaths.sort()

    widths = calculate_object_widths(candidates_fpaths)
    candidates_fpaths, rate = filter_side_img(candidates_fpaths, widths, T=0.5)

    if len(candidates_fpaths) == 1:
        gpt_text_response = "0"
    else:
        prompt_payload = gpt.payload_front_pose(category, candidates_fpaths)
        try:
            gpt_text_response = gpt(payload=prompt_payload, verbose=True)
        except:
            gpt_text_response = gpt(payload=prompt_payload, verbose=True)
        if verbose:
            print(gpt_text_response)
            try:
                print(candidates_fpaths[int(gpt_text_response)])
            except:
                a = 1

    out_dict["front_view"] = candidates_fpaths[int(gpt_text_response)]

    with open(outpath, "w") as f:
        json.dump(out_dict, f, indent=4)

    return True


if __name__ == "__main__":
    objav_path = sys.argv[1]
    category = sys.argv[2]
    print(f"Getting {objav_path} front view ...")
    get_obj_frontview(objav_path, category)
