#!/bin/bash
conda deactivate
cd ~/workspace/SceneWeaver
conda activate infinigen_python


python -m infinigen.launch_blender -m infinigen_examples.generate_indoors --save_dir debug/ -- --seed 0 --task coarse  --output_folder outputs/indoors/coarse_expand_whole_nobedframe -g fast_solve.gin overhead.gin studio.gin -p compose_indoors.terrain_enabled=False

