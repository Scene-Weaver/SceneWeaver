
conda deactivate
cd ~/workspace/SceneWeaver
conda activate infinigen_python

python -m infinigen_examples.generate_indoors --seed 0 --task coarse --output_folder outputs/indoors/coarse_expand_whole_nobedframe -g fast_solve.gin overhead.gin studio.gin -p compose_indoors.terrain_enabled=False compose_indoors.invisible_room_ceilings_enabled=True 
# python -m infinigen_examples.generate_indoors -- --seed 0 --task coarse --output_folder outputs/indoors/coarse_expand_whole_nobedframe -g fast_solve.gin overhead.gin studio.gin -p compose_indoors.terrain_enabled=False
