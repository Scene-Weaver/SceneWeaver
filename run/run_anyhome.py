import json
import subprocess
import time

room_corner_bedroom = [
    [6.2, 14.2],
    [5.6, 11.5],
    [5.9, 6.1],
    [8.3, 10.8],
    [7.4, 12],
    [7.7, 12.4],
    [3.6, 12.9],
    [7.4, 14.1],
    [6.9, 14],
    [3.6, 14.5],
]
# Paths
method = "anyhome"
args_path = f"args_{method}.json"
roominfo_path = f"roominfo_{method}.json"

roomtype = "bedroom"
for idx in range(3, 10):
    # for idx in [11,13,18]:
    print(f"\n=== Running Task {idx} ===")
    orig_json = f"/mnt/fillipo/yandan/scenesage/Anyhome/{roomtype}/{roomtype}_scene_graph(p_center)/anyhome_{roomtype}_scene_graph_{idx}_16.json"
    new_json = f"/mnt/fillipo/yandan/scenesage/Anyhome/{roomtype}/{roomtype}_scene_graph(p_center)/anyhome_{roomtype}_scene_graph_{idx}_16_nobias.json"
    # Edit roominfo_idesign.json
    with open(roominfo_path, "r") as f:
        roominfo_data = json.load(f)

    with open(orig_json, "r") as f:
        j = json.load(f)
        roomsize = j["room_size"]

    for i in range(len(j["objects"])):
        position = j["objects"][i]["position"].copy()
        position["x"] -= round(room_corner_bedroom[idx][0], 4)
        position["y"] = round(room_corner_bedroom[idx][1] - position["y"], 4)
        j["objects"][i]["position"] = position.copy()

    with open(new_json, "w") as f:
        json.dump(j, f, indent=4)

    room_size = [roomsize["width"], roomsize["height"]]
    room_size = [i + 0.28 for i in room_size]
    roominfo_data["roomsize"] = room_size
    roominfo_data["save_dir"] = (
        f"/mnt/fillipo/yandan/scenesage/record_scene/{method}/{roomtype}_{idx}"
    )
    with open(roominfo_path, "w") as f:
        json.dump(roominfo_data, f, indent=4)

    # Edit args_idesign.json
    with open(args_path, "r") as f:
        args_data = json.load(f)
    args_data["json_name"] = new_json
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
