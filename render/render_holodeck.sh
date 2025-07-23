#!/bin/bash

outdir="/mnt/fillipo/yandan/scenesage/record_scene/holodeck"
blend="~/software/blender-4.2.0-linux-x64/blender"
script="~/workspace/SceneWeaver/render/render_single_scene.py"

for roomtype in "$outdir"/*; do
  [ -d "$roomtype" ] || continue
  for room in "$roomtype"/*; do
    [ -d "$room" ] || continue
    [ ! -f "${room}/eevee_idesign_1_view_315.png" ] || continue
    blendfile="$room/record_files/scene_0.blend"
    echo $room
    if [ -f "$blendfile" ]; then
        "$blend" "$blendfile" --background --python "$script" "$room"
    fi
    # break  # Only process the first room
  done
  # break  # Only process the first roomtype
done
