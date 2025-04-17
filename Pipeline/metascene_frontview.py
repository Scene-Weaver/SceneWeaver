import os
import json

from gpt import GPT4
import time

from PIL import Image


def get_object_width(image_path):
    """
    Returns the width of the object in the image by calculating the bounding box of non-transparent pixels.
    """
    # Open the image
    img = Image.open(image_path)
    
    # Convert the image to RGBA (if not already in RGBA)
    img = img.convert("RGBA")
    
    # Get the image data (a list of (r, g, b, a) tuples)
    data = img.getdata()
    
    # Find the bounding box of non-transparent pixels
    left, top, right, bottom = img.width, img.height, 0, 0

    # Iterate through each pixel to find the bounding box of non-transparent regions
    for y in range(img.height):
        for x in range(img.width):
            r, g, b, a = img.getpixel((x, y))
            if a > 0:  # If the pixel is not transparent
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)

    # The width of the object is the difference between right and left boundaries
    object_width = max(right - left,0)
    return object_width


def calculate_object_widths(image_paths):
    """
    Given a list of image paths, returns the largest object width.
    """
    widths = []
    for image_path in image_paths:
        width = get_object_width(image_path)
        widths.append(width)
    
    return widths

def filter_side_img(candidates_fpaths,widths,T=0.3):
    #0,180,270,90
    width_1_max = max(widths[0],widths[1])
    width_1_min = min(widths[0],widths[1])
    width_2_max = max(widths[2],widths[3])
    width_2_min = min(widths[2],widths[3])

    if 0 in widths:
        print("remove empty ",candidates_fpaths[0])
        return [candidates_fpaths[i] for i in range(len(candidates_fpaths)) if widths[i]!=0],1
    
    elif width_1_min*T>width_2_max:
        print("simplify ",candidates_fpaths[0])
        return [candidates_fpaths[0],candidates_fpaths[1]],width_1_min/width_2_max
    
    elif width_2_min*T>width_1_max:
        print("simplify ",candidates_fpaths[0])
        return [candidates_fpaths[2],candidates_fpaths[3]],width_2_min/width_1_max
    
    else:
        return candidates_fpaths,1

def has_front(gpt,category):

    system_prompt = "You will been given an category name. You need to check if the object in this category has a standard 'front view' \n" + \
                    "1. The front view is often characterized by the most significant or most visible face of the object.\n" + \
                    "2. For objects like cabinets, the front view is typically where the doors and drawers are visible. For chairs, the front view may show the seat and backrest. For other objects, consider the main or most notable side visible from the viewer's point of view.\n" + \
                    "3. The front view is usually the view where the object faces the camera directly or is oriented in such a way that the most prominent features (such as a face, label, or handle) are visible.\n" + \
                    "4. Some objects might not have standard front view, such as a bottle, since it looks similar from different angle (front, left, right, and back).\n" + \
                    "Return True if the object usually has a front view, such as desk, transh can, picture, curtain, and monitor. Return False if the object has no front view, such as plant, box, lamp, and object. \n" + \
                    "Example output:True" 
    user_prompt = f"The given category is {category}. Here is your answer (True or False):"
    prompt_payload = gpt.get_payload(system_prompt,user_prompt)
    gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    print(gpt_text_response)
    if gpt_text_response=="True":
        return True
    elif gpt_text_response=="False":
        return False

def get_scene_frontview(scene_name):
    gpt = GPT4(region="swedencentral")
    inbasedir = "/mnt/fillipo/huangyue/recon_sim/7_anno_v4/export_stage2_sm"
    outbasedir = "/mnt/fillipo/yandan/metascene/export_stage2_sm"
    scene_cnt = 0
    obj_cnt = 0
    candidates_fpaths = []
    out_dict = dict()
        
    outinfodir = f"{outbasedir}/{scene_name}/metadata.json"
    
    if not os.path.exists(f"{outbasedir}/{scene_name}"):
        print("missing resource",scene_name)
        return False

    # if os.path.exists(outinfodir):
    #     print("already processed ",scene_name)
    #     return True
    
    print("processing ",scene_name)

    metadata = f"{inbasedir}/{scene_name}/metadata.json"
    scene_cnt += 1
    with open(metadata,"r") as f:
        Placement = json.load(f)

    for key,value in Placement.items():
        time.sleep(1)
        print(f"--- {key} : {value} ---")
        out_dict[key] = {"category":value}
        if value in ["wall","ceiling","floor"]:
            continue
        obj_cnt += 1

        inrenderdir = f"/mnt/fillipo/yandan/metascene/export_stage2_sm/{scene_name}/{key}/"
        candidates_fpaths = []
        for file in os.listdir(inrenderdir):
            candidates_fpaths.append(f"{inrenderdir}/{file}")
        candidates_fpaths.sort()
        
        
    
        widths = calculate_object_widths(candidates_fpaths)
        candidates_fpaths,rate = filter_side_img(candidates_fpaths,widths,T=0.5)

        
        
        if len(candidates_fpaths)==1:
            gpt_text_response = "0"
        elif rate==1 and not has_front(gpt,value):
            gpt_text_response = "0"
        else:
            prompt_payload = gpt.payload_front_pose(value,candidates_fpaths)
            try:
                gpt_text_response = gpt(payload=prompt_payload, verbose=True)
            except:
                gpt_text_response = gpt(payload=prompt_payload, verbose=True)
            print(gpt_text_response)
            try:
                print(candidates_fpaths[int(gpt_text_response)])
            except:
                a = 1

        out_dict[key]["front_view"] = candidates_fpaths[int(gpt_text_response)]
    
    
    with open(outinfodir,"w") as f:
        json.dump(out_dict,f,indent=4)

    return True

if __name__ == "__main__":
    get_scene_frontview("scene0292_00")








