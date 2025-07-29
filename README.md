<div align="center">
<img src="docs/images/sceneweaver.png" width="300"></img>
</div>

<h2 align="center">
  <b>SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent</b>
</h2>
 <div align="center" margin-bottom="6em">
  <a target="_blank" href="https://yandanyang.github.io/">Yandan Yang</a><sup>✶</sup>,
  <a target="_blank" href="https://buzz-beater.github.io/">Baoxiong Jia</a><sup>✶</sup>,
  <a target="_blank" href="https://github.com/hiShujie">Shujie Zhang</a>,
  <a target="_blank" href="https://siyuanhuang.com/">Siyuan Huang</a>

</div>
<br>
<div align="center">
    <!-- <a href="https://cvpr.thecvf.com/virtual/2023/poster/22552" target="_blank"> -->
    <a href="https://arxiv.org/abs/2404.09465" target="_blank"> 
      <img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper arXiv"></a>
    <a href="https://sceneweaver.github.io" target="_blank">
      <img src="https://img.shields.io/badge/Page-SceneWeaver-blue" alt="Project Page"/></a>
</div>
<br>
<div style="text-align: center">
<img src="docs/images/teaser.jpg"  />
</div>


<!-- This is the official repository of [**PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI**](https://arxiv.org/abs/2211.05272). -->


For more information, please visit our [**project page**](https://sceneweaver.github.io).

## Requirements
- Linux machine
- Conda

## ⚙️ Installation & Dependencies

#### Download this repo in your workspace
```
cd ~/workspace
git clone https://github.com/Scene-Weaver/SceneWeaver.git
cd SceneWeaver
```

#### Set LLM api
Save your api-key of GPT in `Pipeline/key.txt`. We use AzureOpenAI here. You can modify this module to fit your own LLM api.

#### Prepare conda env for SceneWeaver's planner:
```
conda env create --prefix /home/yandan/anaconda3/envs/sceneweaver -f environment_sceneweaver.yml
```

#### Prepare conda env for SceneWeaver's executor :
```
conda create --name infinigen python=3.10.14
conda activate infinigen 
pip install bpy==3.6.0
pip install gin-config
pip install frozendict
pip install shapely
pip install trimesh
pip install tqdm
pip install opencv-python
pip install matplotlib
pip install imageio
pip install scipy
pip install scikit-learn
pip install psutil
pip install scikit-image
pip install submitit
pip install python-fcl
pip install pandas
pip install geomdl
pip install Rtree

```

Then, install using one of the options below:
```
# Minimal installation (recommended setting for use in the Blender UI)
INFINIGEN_MINIMAL_INSTALL=True bash scripts/install/interactive_blender.sh

# Normal install
bash scripts/install/interactive_blender.sh

# Enable OpenGL GT
INFINIGEN_INSTALL_CUSTOMGT=True bash scripts/install/interactive_blender.sh
```
More details can refer to [official repo of Infinigen](https://github.com/princeton-vl/infinigen/blob/main/docs/Installation.md#installing-infinigen-as-a-blender-python-script).

                                                        
## Available Tools:

Here we adapt the following tools to our framework. You could choose what you want from the following tools or expand the framework to other tools (such as architecture, Text-2-3D)  you need. 
You should modify `available_tools0` and `available_tools1` in [Pipeline/app/agent/scenedesigner.py](Pipeline/app/agent/scenedesigner.py#L67) to fit the tools you have prepared.

Initializer:
- [x] LLM: GPT
- [x] Dataset: MetaScenes (saved on fillipo)
- [x] Model: PhyScene/DiffuScene/ATISS (we provide some samples in `data/physcene`)
<!-- - [x] Model: PhyScene/DiffuScene/ATISS ([donwload generated scenes in json](https://huggingface.co/datasets/yangyandan/PhyScene/tree/main/generated_scenes)) -->

Implementer:
- [x] Visual: [SD](https://github.com/Scene-Weaver/sd3.5) + [Tabletop Digital Cousin](https://github.com/Scene-Weaver/Tabletop-Digital-Cousins)
- [x] LLM: GPT (both sparse & crowded)
- [x] Rule

Modifier:
- [x] Update Layout/Rotation/Size
- [x] Add Relation
- [x] Remove Objects


## 🛒 Assets      
We here support different source of assets. **You can choose any of them to fit your own requirements**. But in this project, we choose different asset according to the usage of tool.


#### MetaScenes
For tool using Dataset such as MetaScenes, we employ its assets directly, since each scene contains several assets with delicated mesh and layout information.

#### 3D FUTURE
For tool using Model such as PhyScene/DiffuScene/ATISS, we employ 3D FUTURE, since the model is trained on this dataset.
You can download 3D FUTURE in [huggingface](https://huggingface.co/datasets/yangyandan/PhyScene/blob/main/dataset/3D-FUTURE-model.zip).


#### Infinigen 
For other tools, we use [Infinigen's asset generation code](infinigen/assets/objects) to generate standard assets in common categories, such as bed, sofa, and plate. The asset will be generated in a delicated rule procedure in the scene generation process. 

#### Objaverse     
For those catrgories that are not supported by Infinigen, such as clock, laptop, and washing machine, we employ open-vocabulary Objaverse dataset.

We provide two resource & retrieve pipeline for Objaverse (OpenShape & Holodeck), you can following one/both of the two pipelines to retrieve assets. Note if you use **Tabletop Digital Cousin** tool, we recommend you to use Holodeck pipeline.

1. OpenShape
    Refer to [IDesign official repo](https://github.com/atcelen/IDesign/tree/main) and build the `idesign` conda env.
    Run the [inference code](https://github.com/atcelen/IDesign/tree/main?tab=readme-ov-file#inference) to download and build the openshape repo.
    Then run `bash  retrieve.sh debug/`. If success, you will get a new file named`debug/objav_files.json`.

2. Holodeck
    Refer to [Holodeck official repo](https://github.com/allenai/Holodeck?tab=readme-ov-file#data), build the conda env and then download the data.
    Then modify the `ABS_PATH_OF_HOLODECK` in `digital_cousins/models/objaverse/constants.py` to your downloaded directory.
                                          

## Usage

#### Mode 1: Run with Blender in the background
```
cd Pipeline
conda activate sceneweaver
python main.py --prompt "Design me a bedroom." --cnt 1 --basedir PATH/TO/SAVE
```
Then you can check the scene in `PATH/TO/SAVE`. The intermediate scene in each step is saved in `record_files`. You can open relative `.blend` file in blender to check the result of each step.

#### Mode 2: Run with Blender in the foreground
Interactable & convenient to check generating process.

You need to open **two** terminal.

**Terminal 1**: Run infinigen with socket to connect with blender 
```
cd SceneWeaver
conda activate infinigen
python -m infinigen.launch_blender -m infinigen_examples.generate_indoors_vis --save_dir debug/ -- --seed 0 --task coarse  --output_folder debug/ -g fast_solve.gin overhead.gin studio.gin -p compose_indoors.terrain_enabled=False
```
**Terminal 2**: Run SceneWeaver to launch the agent 
```
cd SceneWeaver/Pipeline
conda activate sceneweaver
python main.py --prompt Design me a bedroom. --cnt 1 --basedir PATH/TO/SAVE --socket
```
Then you can check the scene in the `Blender` window and `PATH/TO/SAVE`

#### Generated Folder Structure
We record the intermediate info of each step of the agent and the generated scene. 
The folder structure is as follows:
```
PATH/TO/SAVE/
  Scene_Name/                         # folder name for this scene
    |-- args                          # saved args info for each iter 
      |-- args_{iter}.json
    |-- pipeline                      # saved info for agent
      |-- acdc_output                 # save folder of table top scene
      |-- {tool}_results_{iter}.json  # tool result
      |-- eval_iter_{iter}.json       # eval result
      |-- grade_iter_{iter}.json      # evaluated result (GPT score)
      |-- memory_{iter}.json          # agent memory record
      |-- metric_{iter}.json          # evaluated result (physics & GPT score)
      |-- roomtype.txt                # roomtype
      |-- trajs_{iter}.json           # overall record of previous steps

    |-- record_files                  # record files of intermediate scene
      |-- metric_{iter}.json          # evaluated result (physics)
      |-- name_map_{iter}.json        # name map between object id and blender name
      |-- scene_{iter}.blend          # saved intermediate scene
      |-- obj.blend (optional)        # save supporter for acdc
      |-- env_{iter}.pkl              # record file of infinigen
      |-- house_bbox_{iter}.pkl 
      |-- MaskTag.json
      |-- p_{iter}.pkl
      |-- solved_bbox_{iter}.pkl
      |-- solver_{iter}.pkl
      |-- state_{iter}.pkl
      |-- terrain_{iter}.pkl

    |-- record_scene
      |-- layout_{iter}.json           # object layout & room size
      |-- render_{iter}.jpg            # top-down rendered scene
      |-- render_{iter}_bbox.png       # top-down rendered 3D Mark (bbox, axies, direction, semantic label)
      |-- render_{iter}_marked.jpg     # top-down rendered scene & 3D Mark

    |-- args.json                      # args info for running infinigen 
    |-- objav_cnts.json                # objects to retrieve from objaverse
    |-- objav_files.json               # retrieved results
    |-- roominfo.json                  # room info to start building a new scene
```



## Evaluate 

```
python evaluation_ours.py
```


## Export to USD for Isaac Sim 

```
python -m infinigen.tools.export --input_folder BLENDER_FILE_FOLDER --output_folder USD_SAVE_FOLDER -f usdc -r 1024 --omniverse
```

## 🪧 Citation
If you find our work useful in your research, please consider citing:

```
@inproceedings{yang2024physcene,
          title={PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI},
          author={Yang, Yandan and Jia, Baoxiong and Zhi, Peiyuan and Huang, Siyuan},
          booktitle={Proceedings of Conference on Computer Vision and Pattern Recognition (CVPR)},
          year={2024}
        }
```

## 👋🏻 Acknowledgements
The code of this project is adapted from [Infinigen](https://github.com/princeton-vl/infinigen/tree/main). We sincerely thank the authors for open-sourcing their awesome projects. 