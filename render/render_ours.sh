#!/bin/bash

outdir="/mnt/fillipo/yandan/scenesage/record_scene/manus/"
blend="~/software/blender-4.2.0-linux-x64/blender"
script="~/workspace/SceneWeaver/render/render_single_scene.py"

for roomtype in "$outdir"/*; do
  # roomtype="/mnt/fillipo/yandan/scenesage/record_scene/manus/0_bedroom/"
  [ -d "$roomtype" ] || continue
  for room in "$roomtype"/*; do
    [ -d "$room" ] || continue
    [ ! -f "${room}/eevee_idesign_1_view_315.png" ] || continue
    # echo $room
    # ls $room/record_files/scene_*.blend

    # max_scene=$(ls $room/record_files/scene_*.blend | awk -F '[_.]' 'BEGIN{max=0} {if($(NF-1)>max) max=$(NF-1)} END{print max}')
    max_scene=$(ls "$room/record_files"/scene_*.blend 2>/dev/null | 
        awk -F '[_.]' '
          BEGIN { max=0 }
          $(NF-1) ~ /^[0-9]+$/ { if ($(NF-1) > max) max=$(NF-1) }
          END { print max }
        ')
    echo "The highest scene number is: $max_scene"
    blendfile=${room}/record_files/scene_${max_scene}.blend
    echo $blendfile
    if [ -f "$blendfile" ]; then
        "$blend" "$blendfile" --background --python "$script" "$room"
    fi
    # break  # Only process the first room
  done
  # break  # Only process the first roomtype
done
