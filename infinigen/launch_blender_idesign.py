# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import argparse
import subprocess
from pathlib import Path

root = Path(__file__).parent.parent

BLENDER_BINARY_RELATIVE = [
    root / "blender/blender",
    root / "Blender.app/Contents/MacOS/Blender",
]

# 定义Infinigen导入脚本的路径
IMPORT_INFINIGEN_SCRIPT = root / "infinigen/tools/blendscript_import_infinigen.py"
APPEND_SYSPATH_SCRIPT = root / "infinigen/tools/blendscript_path_append.py"

# 定义Blender无头模式（headless mode）运行时的参数
HEADLESS_ARGS = [
    "-noaudio",  # 禁用音频
    "--background",  # 在后台模式运行
]


# 获取Blender可执行文件的路径
def get_standalone_blender_path():
    try:
        return next(x for x in BLENDER_BINARY_RELATIVE if x.exists())
    except StopIteration:
        raise ValueError(
            "Could not find blender binary - please check you have completed "
            "'Infinigen as a Blender-Python script' section of docs/Installation.md"
            f" and that one of {BLENDER_BINARY_RELATIVE} exists"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--module", type=str, default=None)
    parser.add_argument("-s", "--script", type=str, default=None)
    parser.add_argument("-i", "--iter", type=int, default=None)
    parser.add_argument("--inplace", type=bool, default=False)
    args, unknown_args = parser.parse_known_args()

    import json

    with open("args_idesign.json", "r") as f:
        j = json.load(f)
        args.iter = j["iter"]
        args.inplace = j["inplace"]
    
    import os
    current_file_path = os.path.abspath(__file__)
    with open(f"{current_file_path}/../run/roominfo_idesign.json","r") as f:
        j = json.load(f)
        save_dir = j["save_dir"]
        

    cmd_args = [str(get_standalone_blender_path())]
    if args.inplace:
        cmd_args += [f"{save_dir}/record_files/scene_{args.iter}.blend"]
    else:
        cmd_args += [f"{save_dir}/record_files/scene_{args.iter-1}.blend"]
    if args.module is not None:
        # cmd_args += HEADLESS_ARGS

        # 追加路径脚本（用于在Blender中追加Python路径）
        cmd_args += ["--python", str(APPEND_SYSPATH_SCRIPT)]
        # 将模块路径转换为Blender Python脚本路径
        relpath = "/".join(args.module.split(".")) + ".py"
        path = root / relpath
        if not path.exists():  # 如果指定的Python脚本不存在，抛出文件未找到异常
            raise FileNotFoundError(f"Could not find python script {path}")
        # 将模块脚本加入Blender运行的命令行参数
        cmd_args += ["--python", str(path)]

    elif args.script is not None:  # 如果指定了脚本参数
        # 启动无头模式并指定要运行的Python脚本
        cmd_args += HEADLESS_ARGS + ["--python", args.script]
    else:  # 如果没有指定模块或脚本，使用默认的Infinigen导入脚本
        cmd_args += ["--python", str(IMPORT_INFINIGEN_SCRIPT)]

    if len(unknown_args):
        cmd_args += unknown_args

    print(" ".join(cmd_args))

    # 使用subprocess.run()运行命令

    subprocess.run(cmd_args, cwd=root)  # 在root目录下执行Blender


# python -m infinigen.launch_blender -m infinigen_examples.generate_indoors -- --seed 0 --task coarse --output_folder outputs/indoors/coarse -g fast_solve.gin overhead.gin singleroom.gin -p compose_indoors.terrain_enabled=False compose_indoors.overhead_cam_enabled=True compose_indoors.solve_max_rooms=1 compose_indoors.invisible_room_ceilings_enabled=True compose_indoors.restrict_single_supported_roomtype=True
# python -m infinigen.launch_blender -m match.__init__ --
