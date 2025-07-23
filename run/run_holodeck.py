import json
import os
import subprocess
import time

# Paths
method = "holodeck"
args_path = f"args_{method}.json"
roominfo_path = f"roominfo_{method}.json"

# roomtype = "bedroom"
basedir = "/mnt/fillipo/yandan/scenesage/Holodeck0509/"
outdir = "/mnt/fillipo/yandan/scenesage/record_scene/holodeck/"
roomtypes = os.listdir(basedir)
for roomtype in roomtypes:
    if roomtype not in ["office"]:
        # if roomtype not in ["bathroom","children_room","classroom","gameroom","gym","kitchen","laundromat","meetingroom","office","restaurant","waitingroom"]:
        continue
    os.makedirs(f"{outdir}/{roomtype}/", exist_ok=True)
    for idx in range(3):
        # for idx in [11,13,18]:
        print(f"\n=== Running Task {idx} ===")
        jsonname = f"{basedir}/{roomtype}/{roomtype}_{idx}.json"
        print(f"!!!!processing {jsonname}")
        # Edit roominfo_idesign.json
        with open(roominfo_path, "r") as f:
            roominfo_data = json.load(f)

        with open(jsonname, "r") as f:
            j = json.load(f)
            roomsize = j["room_size"]

        room_size = [roomsize["length"], roomsize["width"]]
        room_size = [i + 0.28 for i in room_size]
        roominfo_data["roomsize"] = room_size
        roominfo_data["save_dir"] = f"{outdir}/{roomtype}/{roomtype}_{idx}"
        if os.path.exists(roominfo_data["save_dir"]):
            continue
        with open(roominfo_path, "w") as f:
            json.dump(roominfo_data, f, indent=4)

        # Edit args_idesign.json
        with open(args_path, "r") as f:
            args_data = json.load(f)
        args_data["json_name"] = jsonname
        with open(args_path, "w") as f:
            json.dump(args_data, f, indent=4)

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
            f"{method}",
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
