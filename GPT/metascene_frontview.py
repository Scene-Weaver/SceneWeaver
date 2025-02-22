import os
import json

from gpt import GPT4 as gpt
gpt = gpt()
inbasedir = "/mnt/fillipo/huangyue/recon_sim/7_anno_v4/export_stage2_sm"
outbasedir = "/mnt/fillipo/yandan/metascene/export_stage2_sm"
scene_cnt = 0
obj_cnt = 0
candidates_fpaths = []
out_dict = dict()
for scene_name in os.listdir(inbasedir):
    scene_name = "scene0001_00"
    metadata = f"{inbasedir}/{scene_name}/metadata.json"
    scene_cnt += 1
    with open(metadata,"r") as f:
        Placement = json.load(f)


    for key,value in Placement.items():
        out_dict[key] = {"category":value}
        if value in ["wall","ceiling","floor"]:
            continue
        obj_cnt += 1

        inrenderdir = f"/mnt/fillipo/yandan/metascene/export_stage2_sm/{scene_name}/{key}/"
        candidates_fpaths = []
        for file in os.listdir(inrenderdir):
            candidates_fpaths.append(f"{inrenderdir}/{file}")

        prompt_payload = gpt.payload_front_pose(value,candidates_fpaths)
        gpt_text_response = gpt(payload=prompt_payload, verbose=True)
        print(gpt_text_response)

        out_dict[key]["front_view"] = candidates_fpaths[int(gpt_text_response)]
    
    outinfodir = f"{outbasedir}/{scene_name}/metadata.json"
    with open(outinfodir,"w") as f:
        json.dump(out_dict,f,indent=4)
    break

# print(obj_cnt,obj_cnt/scene_cnt)













