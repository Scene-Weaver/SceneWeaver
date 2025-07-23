#!/usr/bin/env python

import json
import multiprocessing
import os
import re
import shutil
import sys
import threading

import numpy as np
import objaverse
import openshape
import torch
import transformers
from huggingface_hub import hf_hub_download
from torch.nn import functional as F

# Print device
print("Device: ", torch.cuda.get_device_name(0))

# Load the Pointcloud Encoder
pc_encoder = openshape.load_pc_encoder("openshape-pointbert-vitg14-rgb")

# Get the pre-computed embeddings
meta = json.load(
    open(
        hf_hub_download(
            "OpenShape/openshape-objaverse-embeddings",
            "objaverse_meta.json",
            token=True,
            repo_type="dataset",
            local_dir="~/workspace/IDesign/OpenShape-Embeddings",
        )
    )
)

meta = {x["u"]: x for x in meta["entries"]}
deser = torch.load(
    hf_hub_download(
        "OpenShape/openshape-objaverse-embeddings",
        "objaverse.pt",
        token=True,
        repo_type="dataset",
        local_dir="~/workspace/IDesign/OpenShape-Embeddings",
    ),
    map_location="cpu",
)
us = deser["us"]
feats = deser["feats"]


def move_files(file_dict, destination_folder, id):
    os.makedirs(destination_folder, exist_ok=True)
    for item_id, file_path in file_dict.items():
        destination_path = f"{destination_folder}{id}.glb"

        shutil.move(file_path, destination_path)
        print(f"File {item_id} moved from {file_path} to {destination_path}")


def load_openclip():
    print("Locking...")
    sys.clip_move_lock = threading.Lock()
    print("Locked.")
    clip_model, clip_prep = (
        transformers.CLIPModel.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            low_cpu_mem_usage=True,
            torch_dtype=half,
            offload_state_dict=True,
        ),
        transformers.CLIPProcessor.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        ),
    )
    if torch.cuda.is_available():
        with sys.clip_move_lock:
            clip_model.cuda()
    return clip_model, clip_prep


def retrieve(embedding, top, sim_th=0.0, filter_fn=None):
    sims = []
    embedding = F.normalize(embedding.detach().cpu(), dim=-1).squeeze()
    for chunk in torch.split(feats, 10240):
        sims.append(embedding @ F.normalize(chunk.float(), dim=-1).T)
    sims = torch.cat(sims)
    sims, idx = torch.sort(sims, descending=True)
    sim_mask = sims > sim_th
    sims = sims[sim_mask]
    idx = idx[sim_mask]
    results = []
    for i, sim in zip(idx, sims):
        if us[i] in meta:
            if filter_fn is None or filter_fn(meta[us[i]]):
                results.append(dict(meta[us[i]], sim=sim))
                if len(results) >= top:
                    break
    return results


def get_filter_fn():
    face_min = 0
    face_max = 34985808
    anim_min = 0
    anim_max = 563
    anim_n = not (anim_min > 0 or anim_max < 563)
    face_n = not (face_min > 0 or face_max < 34985808)
    filter_fn = lambda x: (
        (anim_n or anim_min <= x["anims"] <= anim_max)
        and (face_n or face_min <= x["faces"] <= face_max)
    )
    return filter_fn


def preprocess(input_string):
    wo_numericals = re.sub(r"\d", "", input_string)
    output = wo_numericals.replace("_", " ")
    return output


if __name__ == "__main__":
    
    save_dir = sys.argv[1]

    # if not os.path.exists(f"{save_dir}/objav_files.json"):
    if True:
        with open(f"{save_dir}/objav_cnts.json", "r") as f:
            LoadObjavCnts = json.load(f)

        f32 = np.float32
        half = torch.float16 if torch.cuda.is_available() else torch.bfloat16
        clip_model, clip_prep = load_openclip()
        torch.set_grad_enabled(False)

        LoadObjavFiles = dict()
        if not os.path.exists(f"{save_dir}/objav_files.json"):
            for category, cnt in LoadObjavCnts.items():
                if category.lower() == "meeting_table":
                    a = 1
                if category.lower() == "car":
                    LoadObjavFiles["car"] = [
                        "~/.objaverse/hf-objaverse-v1/glbs/000-068/45840e2136c44080b4c1e7521cce8db3.glb"
                    ]
                    continue
                # text = preprocess(f"A high-poly {roomtype} {category} in high quality")
                text = preprocess(f"A high-poly realistic {category} in high quality")
                device = clip_model.device
                tn = clip_prep(
                    text=[text], return_tensors="pt", truncation=True, max_length=76
                ).to(device)
                LoadObjavFiles[category] = []
                enc = clip_model.get_text_features(**tn).float().cpu()
                retrieved_objs = retrieve(
                    enc, top=10, sim_th=0.1, filter_fn=get_filter_fn()
                )
                # import pdb
                # pdb.set_trace()
                for i in range(len(retrieved_objs)):
                    retrieved_obj = retrieved_objs[i]
                    if retrieved_obj["u"] in [
                        "df9af4b3c2ea40d89a736741e8c07bb1",
                        "75e4d132d8e5480e99f915f0464aeff0",
                        "45a2ad85a21d46fabfe38d492ed3ec04",
                        "90aae32de40c458e846a3705105e5cad",
                        "9d2946e980354264bf6be4a41f21f81e",
                        "d7403315f4934dbd913578dc32f1962f",
                        "6cbddf0c4c5a4cacad14c6a8fa94f22c",
                        "907649f7c56e478dac505f91318f59cc",
                        "6bdcf3960b434396b5a194f6685e2cbc",
                        "0d3dd9d37ada4153b82c92e1fb4d4c4f",
                        "f3d62c8081994608a8b25b6db083cb1c",
                        "bf375674c14f4ab8a0a16aad8cc99bab",
                    ]:
                        continue
                    print("Retrieved object: ", retrieved_obj["u"])
                    processes = multiprocessing.cpu_count()
                    try:
                        objaverse_objects = objaverse.load_objects(
                            uids=[retrieved_obj["u"]], download_processes=processes
                        )
                    except:
                        continue
                    file_path = list(objaverse_objects.values())[0]

                    LoadObjavFiles[category].append(file_path)
                    if len(LoadObjavFiles[category]) >= 3:
                        break

            with open(f"{save_dir}/objav_files.json", "w") as f:
                json.dump(LoadObjavFiles, f, indent=4)
