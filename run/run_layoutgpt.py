import json
import os
import subprocess
import time

# Paths
args_path = "args_layoutgpt.json"
roominfo_path = "roominfo_layoutgpt.json"

basedir = "/mnt/fillipo/yandan/scenesage/LayoutGPT05133/"
outdir = "/mnt/fillipo/yandan/scenesage/record_scene/layoutgpt/"
# roomtype = "livingroom"
roomtypes = os.listdir(basedir)
for roomtype in roomtypes:  # ["bedroom","livingroom"]:
    if roomtype not in [
        "meetingroom",
        "office",
        "restaurant",
        "waitingroom",
        "bathroom",
        "children_room",
        "gym",
        "kitchen",
    ]:
        continue
    # if roomtype not in ["garage"]:
    #     continue
    os.makedirs(f"{outdir}/{roomtype}/", exist_ok=True)
    for i in range(1, 4):
        # for i in [11,13,18]:

        # Edit args_idesign.json
        with open(args_path, "r") as f:
            args_data = json.load(f)
        # args_data["json_name"] = (
        #     f"/mnt/fillipo/yandan/scenesage/LayoutGPT/{roomtype}_scene_graph/layoutgpt_{roomtype}_{i}.json"
        # )
        args_data["json_name"] = f"{basedir}/{roomtype}/{roomtype}_%02d.json" % i
        json_name = args_data["json_name"]
        print(f"\n=== Running Task {json_name}===")
        with open(args_path, "w") as f:
            json.dump(args_data, f, indent=4)

        # Edit roominfo_idesign.json
        with open(roominfo_path, "r") as f:
            roominfo_data = json.load(f)

        with open(args_data["json_name"], "r") as f:
            j = json.load(f)
            # roomsize = j["room_size"]
            roomsize = j["roomsize"]

        room_size = [roomsize["width"], roomsize["length"]]
        room_size = [i + 0.28 for i in room_size]
        roominfo_data["roomsize"] = room_size
        roominfo_data["save_dir"] = f"{outdir}/{roomtype}/{roomtype}_{i}"
        if os.path.exists(roominfo_data["save_dir"]):
            continue
        with open(roominfo_path, "w") as f:
            json.dump(roominfo_data, f, indent=4)

        # Run the Blender generation command
        command = [
            "python",
            # "-m", "infinigen.launch_blender",
            "-m",
            "infinigen_examples.generate_indoors_idesign",
            # "--",
            "--seed",
            "0",
            "--task",
            "coarse",
            "--method",
            "layoutgpt",
            "--output_folder",
            "outputs/indoors/coarse_expand_whole_nobedframe",
            "-g",
            "fast_solve.gin",
            "overhead.gin",
            "studio.gin",
            "-p",
            "compose_indoors.terrain_enabled=False",
        ]
        subprocess.run(command)

        time.sleep(2)  # Optional: wait a bit between tasks

    print("\n🎯 All tasks finished!")
