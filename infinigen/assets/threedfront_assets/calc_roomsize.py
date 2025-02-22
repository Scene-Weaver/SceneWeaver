import os
import json
import bpy
import mathutils

metadata = "/mnt/fillipo/yandan/metascene/export_stage2_sm/scene0001_00/metadata.json"


basedir = "/mnt/fillipo/huangyue/recon_sim/7_anno_v4/export_stage2_sm"
outbasedir = "/mnt/fillipo/yandan/metascene/export_stage2_sm/"

record = dict()

idx = 0
for scene_name in sorted(os.listdir(basedir)):
    idx += 1
    print(f"################## processing idx {idx} : {scene_name} ######################")
    
    metadata = f"{basedir}/{scene_name}/metadata.json"


    with open(metadata,"r") as f:
        Placement = json.load(f)
    for key,value in Placement.items():
        category = value
        
        if category == "floor":
            if scene_name in record:
                a = 1

            bpy.ops.import_scene.gltf(filepath=f"{basedir}/{scene_name}/{key}.glb")
            imported_obj = bpy.context.selected_objects[0]
            bbox_local = imported_obj.bound_box
            # Convert the bounding box corners to world space by applying the object's transformation
            bbox_world = [imported_obj.matrix_world @ mathutils.Vector(corner) for corner in bbox_local]

            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')

            # Iterate over the transformed corners to find min and max values for x and y
            for corner in bbox_world:
                min_x = min(min_x, corner.x)
                max_x = max(max_x, corner.x)
                min_y = min(min_y, corner.y)
                max_y = max(max_y, corner.y)

            size_x = max_x - min_x
            size_y = max_y - min_y

            roomsize = {
                        "min_x":min_x,
                        "max_x":max_x,
                        "min_y":min_y,
                        "max_y":max_y,
                        "size_x":size_x,
                        "size_y":size_y,
                    }
            
            record[scene_name] = roomsize



statisticdir =  f"{outbasedir}/roomsize.json"
with open(statisticdir,"w") as f:
    json.dump(record,f,indent=4)