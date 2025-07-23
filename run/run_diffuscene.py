import json
import os
import random
import subprocess
import time

import numpy as np


def calculate_room_size(data):
    min_coords = np.array([float("inf"), float("inf"), float("inf")])
    max_coords = np.array([-float("inf"), -float("inf"), -float("inf")])

    for objects in data["ThreedFront"].values():
        for obj in objects:
            position = np.array(obj["position"])  # Object's position
            size = np.array(obj["size"])  # Half-size for bounding box calculation

            # Compute object's bounding box min and max coordinates
            obj_min = position - size
            obj_max = position + size

            # Update overall min/max coordinates
            min_coords = np.minimum(min_coords, obj_min)
            max_coords = np.maximum(max_coords, obj_max)

    # Calculate room size (max - min)
    # room_size = max_coords - min_coords
    room_size = 2 * np.maximum(abs(max_coords), abs(min_coords))
    return room_size[0], room_size[2]


# Paths
method = "diffuscene"
args_path = f"args_{method}.json"
roominfo_path = f"roominfo_{method}.json"

# roomtype = "livingroom"
for roomtype in ["bedroom", "livingroom"]:
    basedir = "data/physcene"
    files = os.listdir(basedir)
    files = [i for i in files if roomtype in i and i.endswith(".json")]
    random.shuffle(files)
    for i in range(3):
        json_name = f"{basedir}/{files[i]}"
        print(f"\n=== Running Task {i} ===")

        # Edit args_idesign.json
        with open(args_path, "r") as f:
            args_data = json.load(f)
            args_data["json_name"] = json_name

        with open(args_path, "w") as f:
            json.dump(args_data, f, indent=4)

        # Edit roominfo_idesign.json
        with open(roominfo_path, "r") as f:
            roominfo_data = json.load(f)

        with open(json_name, "r") as f:
            data = json.load(f)
            room_size = calculate_room_size(data)

        room_size = [round(i + 0.28, 3) for i in room_size]
        roominfo_data["roomsize"] = room_size
        roominfo_data["save_dir"] = (
            f"/mnt/fillipo/yandan/scenesage/record_scene/{method}/{roomtype}_{i}"
        )
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
