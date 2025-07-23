#!/bin/bash
conda deactivate
cd ~/workspace/infinigen
conda activate infinigen_python

# blender viewport
python -m infinigen.launch_blender -m infinigen_examples.generate_indoors_idesign -- --seed 0 --task coarse --output_folder outputs/indoors/coarse_expand_whole_nobedframe -g fast_solve.gin overhead.gin studio.gin -p compose_indoors.terrain_enabled=False
