import os  
import json
from gpt import GPT4 as gpt
from prompt_room import extract_json, dict2str

def json2sentence(categories):
    info = []
    for cat, num in categories.items():
        info.append(str(num)+" "+cat)
    s = ", ".join(info[:-1])+", and " + info[-1]
    return s

def convert_roomtype(gpt_text_response, roomtype_cnt):
    result = []
    roomtypes = []
    for info in gpt_text_response.split(";"):
        roomtype,score = info.split(",")
        roomtypes.append(roomtype)
        result.append({"predicted":roomtype.strip(), "confidence": float(score.strip())})
        
    roomtypes = ", ".join(roomtypes)
    if roomtypes not in roomtype_cnt:
        roomtype_cnt[roomtypes] = 0
    roomtype_cnt[roomtypes] += 1

    return result, roomtype_cnt

gpt = gpt()

basedir = "/mnt/fillipo/huangyue/recon_sim/7_anno_v4/export_stage2_sm"
outbasedir = "/mnt/fillipo/yandan/metascene/export_stage2_sm/"

# record = dict()
# record["scenes"] = dict()
category_set = set()
roomtype_cnt = dict()
# idx = 0
# for scene_name in sorted(os.listdir(basedir)):
#     idx += 1
#     if idx<=328:
#         continue
#     print(f"################## processing idx {idx} : {scene_name} ######################")
#     record["scenes"][scene_name] = dict()
#     record["scenes"][scene_name]["categories"] = dict()
#     metadata = f"{basedir}/{scene_name}/metadata.json"
#     with open(metadata,"r") as f:
#         Placement = json.load(f)  
#     if len(Placement.keys())==0:
#         continue
#     for key,value in Placement.items():
#         category = value
#         if category in ["wall","ceiling","floor"]:
#             continue
#         if category not in record["scenes"][scene_name]["categories"]:
#             record["scenes"][scene_name]["categories"][category] = 0
#         record["scenes"][scene_name]["categories"][category] += 1
        

#     obj_cnts = json2sentence(record["scenes"][scene_name]["categories"])
#     prompt_payload = gpt.payload_roomtype(obj_cnts)
#     gpt_text_response = gpt(payload=prompt_payload, verbose=True)
#     roomtype_info, roomtype_cnt = convert_roomtype(gpt_text_response, roomtype_cnt)
#     print(roomtype_info)
#     record["scenes"][scene_name]["roomtype"] = roomtype_info
    

#     obj_info = dict2str(record["scenes"][scene_name]["categories"])
#     prompt_payload = gpt.payload_simplify_cnts(obj_info)
#     gpt_text_response = gpt(payload=prompt_payload, verbose=True)
#     print(gpt_text_response)
#     obj_info_simplified = extract_json(gpt_text_response)
#     record["scenes"][scene_name]["categories_gpt_merged"] = obj_info_simplified

#     category_set = category_set.union(set(obj_info_simplified.keys()))
    
#     if idx%10==1:
#         statisticdir =  f"{outbasedir}/statistic.json"
#         with open(statisticdir,"w") as f:
#             json.dump(record,f,indent=4)

#     # print(f"################## finished idx {idx} : {scene_name} ######################")

# statisticdir =  f"{outbasedir}/statistic.json"
# with open(statisticdir,"w") as f:
#     json.dump(record,f,indent=4)

statisticdir =  f"{outbasedir}/statistic.json"
with open(statisticdir,"r") as f:
    record = json.load(f)

scenes = record["scenes"]
for scenename, sceneinfo in scenes.items():
    if len(sceneinfo["categories"])==0:
        continue
    for roomtype in sceneinfo["roomtype"]:
        name = roomtype["predicted"]
        if name not in roomtype_cnt:
            roomtype_cnt[name] = 0
        roomtype_cnt[name] += 1
    categories = set(sceneinfo["categories_gpt_merged"].keys())
    category_set = category_set.union(categories)

print(roomtype_cnt)
roomtypedir =  f"{outbasedir}/roomtype.json"
with open(roomtypedir,"w") as f:
    json.dump(roomtype_cnt,f,indent=4)  


print(category_set)
categorydir =  f"{outbasedir}/category.json"
with open(categorydir,"w") as f:
    json.dump(list(category_set),f,indent=4)  