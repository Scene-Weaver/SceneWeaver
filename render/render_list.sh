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

# /mnt/fillipo/yandan/scenesage/record_scene/layoutgpt/restaurant/restaurant_2/
# /mnt/fillipo/yandan/scenesage/record_scene/layoutgpt/meetingroom/meetingroom_1/
# /mnt/fillipo/yandan/scenesage/record_scene/layoutgpt/kitchen/kitchen_2/
# /mnt/fillipo/yandan/scenesage/record_scene/layoutgpt/office/office_1/

# /mnt/fillipo/yandan/scenesage/record_scene/idesign/restaurant/scene_3/
# /mnt/fillipo/yandan/scenesage/record_scene/idesign/meetingroom/scene_0/
# /mnt/fillipo/yandan/scenesage/record_scene/idesign/office/scene_3/
# /mnt/fillipo/yandan/scenesage/record_scene/idesign/office/scene_4/
# /mnt/fillipo/yandan/scenesage/record_scene/idesign/kitchen/scene_0/

# /mnt/fillipo/yandan/scenesage/record_scene/holodeck/meetingroom/meetingroom_1/
# /mnt/fillipo/yandan/scenesage/record_scene/holodeck/office/office_1/
# /mnt/fillipo/yandan/scenesage/record_scene/holodeck/restaurant/restaurant_0/
# /mnt/fillipo/yandan/scenesage/record_scene/holodeck/kitchen/kitchen_1/

# /mnt/fillipo/yandan/scenesage/record_scene/manus/0_restaurant/Design_me_a_single_room_of_res_0/
# /mnt/fillipo/yandan/scenesage/record_scene/manus/0_meeting_room/Design_me_a_meeting_room_0/
# /mnt/fillipo/yandan/scenesage/record_scene/manus/0_meeting_room/Design_me_a_meeting_room_2_good/
# /mnt/fillipo/yandan/scenesage/record_scene/manus/0_kitchen/Design_me_a_kitchen_0/
# /mnt/fillipo/yandan/scenesage/record_scene/manus/0_office/Design_me_an_office_4/
# /mnt/fillipo/yandan/scenesage/record_scene/manus/0_kitchen_nomodifier/Design_me_a_kitchen_2/
# /mnt/fillipo/yandan/scenesage/record_scene/manus/0_office/Design_me_an_office_2/
# /mnt/fillipo/yandan/scenesage/record_scene/manus/0_restaurant/Design_me_a_restaurant_room_2/record_scene/

/mnt/fillipo/yandan/scenesage/record_scene/atiss/bedroom/bedroom_2/
/mnt/fillipo/yandan/scenesage/record_scene/atiss/livingroom/livingroom_0/
/mnt/fillipo/yandan/scenesage/record_scene/diffuscene/bedroom/bedroom_5/
/mnt/fillipo/yandan/scenesage/record_scene/diffuscene/livingroom/livingroom_1/

/mnt/fillipo/yandan/scenesage/record_scene/physcene/livingroom/livingroom_1/
/mnt/fillipo/yandan/scenesage/record_scene/physcene/livingroom/livingroom_0/

/mnt/fillipo/yandan/scenesage/record_scene/manus/0_bedroom/Design_me_a_bedroom_10_good/
/mnt/fillipo/yandan/scenesage/record_scene/manus/0_bedroom/Design_me_a_bedroom_13/
/mnt/fillipo/yandan/scenesage/record_scene/manus/0_bedroom/Design_me_a_bedroom_8_good/

/mnt/fillipo/yandan/scenesage/record_scene/manus/0_livingroom/Design_me_a_living_room_1/
/mnt/fillipo/yandan/scenesage/record_scene/manus/0_livingroom/Design_me_a_living_room_8/
/mnt/fillipo/yandan/scenesage/record_scene/manus/0_livingroom/Design_me_a_living_room_2_good/