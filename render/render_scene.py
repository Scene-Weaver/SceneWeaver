import os

outdir = "/mnt/fillipo/yandan/scenesage/record_scene/holodeck/"
roomtypes = os.listdir(outdir)
blend = "~/software/blender-4.2.0-linux-x64/blender"
for roomtype in roomtypes:
    for roomname in os.listdir(f"{outdir}/{roomtype}/"):
        roomdir = f"{outdir}/{roomtype}/{roomname}"
        blendfile = f"{roomdir}/record_files/scene_0.blend"
        os.system(
            f"{blend} {blendfile} --background --python ~/workspace/infinigen/render/render_scene.py"
        )
        break
    break
