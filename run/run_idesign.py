import json
import os
import subprocess
import time

# Room sizes for each task (replace with your actual room sizes)
# room_sizes = [
#     # bedroom
#     [3.0, 4.0, 2.4],
#     [2.5, 3.0, 2.4],
#     [3.5, 4.5, 2.4],
#     [4.0, 5.0, 2.4],
#     [2.4, 3.5, 2.4],
#     [3.2, 4.2, 2.4],
#     [2.8, 3.6, 2.4],
#     [3.6, 4.8, 2.4],
#     [4.2, 5.2, 2.4],
#     [3.0, 3.5, 2.4],
#     # living room
#     [4.0, 5.0, 2.8],
#     [3.5, 4.5, 2.8],
#     [3.0, 4.0, 2.8],
#     [4.5, 6.0, 3.0],
#     [5.0, 7.0, 3.0],
#     [3.6, 4.8, 2.8],
#     [4.2, 5.2, 2.8],
#     [5.5, 6.5, 3.0],
#     [3.2, 4.2, 2.8],
#     [6.0, 8.0, 3.0],
# ]
with open("run_idesign_roomsize.json", "r") as f:
    roomsizes = json.load(f)
# Paths
args_path = "args_idesign.json"
roominfo_path = "roominfo_idesign.json"
basedir = "/mnt/fillipo/yandan/scenesage/idesign0509/"
outdir = "/mnt/fillipo/yandan/scenesage/record_scene/idesign"
for roomtype in ["classroom"]:
    os.makedirs(f"{outdir}/{roomtype}/", exist_ok=True)
    for i in range(1, 5):
        # for i in [11,13,18]:
        print(f"\n=== Running Task {i} ===")

        # Edit args_idesign.json
        with open(args_path, "r") as f:
            args_data = json.load(f)
        args_data["json_name"] = (
            f"{basedir}/{roomtype}/scene_graph/{roomtype}_scene_graph_{i}.json"
        )
        if not os.path.exists(args_data["json_name"]):
            args_data["json_name"] = (
                f"{basedir}/{roomtype}/scene_graph/{roomtype}_graph_{i}.json"
            )
        if not os.path.exists(args_data["json_name"]):
            continue

        with open(args_path, "w") as f:
            json.dump(args_data, f, indent=4)

        # Edit roominfo_idesign.json
        with open(roominfo_path, "r") as f:
            roominfo_data = json.load(f)

        room_size = roomsizes[roomtype][i][:2]
        room_size = [i + 0.28 for i in room_size]
        roominfo_data["roomsize"] = room_size
        roominfo_data["save_dir"] = f"{outdir}/{roomtype}/scene_{i}"
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
            "idesign",
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
        # subprocess.run(["bash", "-c", command])

        time.sleep(2)  # Optional: wait a bit between tasks
        # except:
        #     print("error processing room.")

    print("\n🎯 All tasks finished!")
