import math

import open_clip
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from .objaverse_retriever import ObjathorRetriever


class ObjectRetriever:
    def __init__(self, objaverse_version="2023_09_23"):
        self.retrieval_threshold = 28
        self.clip_threshold = 30
        self.used_assets = []
        self.objaverse_version = objaverse_version

        self.clip_model, _, self.clip_preprocess = (
            open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="laion2b_s32b_b82k"
            )
        )
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.sbert_model = SentenceTransformer("all-mpnet-base-v2")

        self.retrieval_threshold = 28
        self.object_retriever = ObjathorRetriever(
            clip_model=self.clip_model,
            clip_preprocess=self.clip_preprocess,
            clip_tokenizer=self.clip_tokenizer,
            sbert_model=self.sbert_model,
            retrieval_threshold=self.retrieval_threshold,
        )

        self.database = self.object_retriever.database
        # self.check = self.object_retriever.check
        self.json_template = {
            "assetId": None,
            "id": None,
            "label": None,
            "scale": [1, 1, 1],
            "theta": 0,
            "kinematic": False,
            "position": {},
            "rotation": {},
            "material": None,
        }

    def retrieve_small_objs(self, scene):
        def convert_format(results):
            for i in range(len(results)):
                results[i]["position"] = list(results[i]["position"].values())
                results[i]["size"] = list(results[i]["size"].values())
                results[i]["rotation"] = list(results[i]["rotation"].values())
                results[i]["theta"] = results[i]["theta"] * math.pi / 180
            return results

        for dataset in ["ThreedFront", "GPN"]:
            for assetname in scene[dataset]:
                assets = scene[dataset][assetname]
                for i in range(len(assets)):
                    idx = scene[dataset][assetname][i]["index"]
                    large_obj = scene[dataset][assetname][i]
                    ordered_small_objects = self.select_small_objects_per_receptacle(
                        large_obj
                    )
                    results = self.place_small_objects(large_obj, ordered_small_objects)
                    results = convert_format(results)
                    scene[dataset][assetname][i]["objects_on_top"] = results
        return scene

    def place_object(self, asset_id, large_obj, rotation):
        obj = dict()
        # large object
        position = large_obj["position"]
        centroid = large_obj["centroid"]
        top_center = [
            position[0] + centroid[0],
            position[1] + centroid[1],
            position[2] + centroid[2],
        ]

        top_center[1] += large_obj["scale"][1] * large_obj["size"][1]

        obj["position"] = {"x": top_center[0], "y": top_center[1], "z": top_center[2]}
        # # small object
        # center_position = self.database[asset_id]["objectMetadata"]["axisAlignedBoundingBox"]["center"].copy()
        # dimensions = self.database[asset_id]["assetMetadata"]["boundingBox"]
        # place_position = [top_center[0],top_center[1]+center_position["y"],top_center[2]]
        # obj["position_correct"] = {
        #     "x": place_position[0],
        #     "y": place_position[1],
        #     "z": place_position[2]
        # }

        # receptacle_dimensions = large_obj["size"]
        # receptacle_size = [receptacle_dimensions[0], receptacle_dimensions[1]]
        # x_rate = custom_distribution()
        # y_rate = custom_distribution()
        # sorted(receptacle_size)
        # receptacle_area = receptacle_size[0] * receptacle_size[1]
        # capacity = 0

        return obj

    def fix_placement_for_thin_assets(self, placement):
        asset_id = placement["assetId"]
        dimensions = self.database[asset_id]["assetMetadata"]["boundingBox"]
        threshold = 0.03  # 0.03 meter is the threshold for thin objects

        orginal_rotation = placement["rotation"]
        orginal_position = placement["position"]
        bottom_center_position = {
            "x": orginal_position["x"],
            "y": orginal_position["y"] - dimensions["y"] / 2,
            "z": orginal_position["z"],
        }

        if dimensions["x"] <= threshold:
            # asset is thin in x direction, need to rotate in z direction
            placement["rotation"] = {
                "x": orginal_rotation["x"],
                "y": orginal_rotation["y"],
                "z": orginal_rotation["z"] + 90,
            }
            placement["position"] = {
                "x": bottom_center_position["x"],
                "y": bottom_center_position["y"] + dimensions["x"] / 2,
                "z": bottom_center_position["z"],
            }

        elif dimensions["z"] <= threshold:
            # asset is thin in z direction, need to rotate in x direction
            placement["rotation"] = {
                "x": orginal_rotation["x"] + 90,
                "y": orginal_rotation["y"],
                "z": orginal_rotation["z"],
            }
            placement["position"] = {
                "x": bottom_center_position["x"],
                "y": bottom_center_position["y"] + dimensions["z"] / 2,
                "z": bottom_center_position["z"],
            }

        return placement

    def place_small_objects(self, large_obj, small_objects):
        # TODO Check xyz order
        wall_height = 200
        results = []
        # Place the objects
        placements = []
        for object_name, asset_id, _ in small_objects:
            thin, rotation = self.check.check_thin_asset(asset_id)
            small, y_rotation = self.check.check_small_asset(
                asset_id
            )  # check if the object is small and rotate around y axis randomly

            obj = self.place_object(asset_id, large_obj, rotation)

            if obj != None:  # If the object is successfully placed
                placement = self.json_template.copy()
                placement["assetId"] = asset_id
                placement["id"] = f"{object_name}"
                placement["label"] = object_name.split("-")[0]
                placement["scale"] = [1, 1, 1]
                placement["theta"] = y_rotation

                placement["position"] = obj["position"]
                # placement["position_correct"] = obj["position_correct"]
                placement["rotation"] = {"x": 0, "y": y_rotation, "z": 0}
                placement["size"] = self.database[asset_id]["assetMetadata"][
                    "boundingBox"
                ]
                asset_height = self.database[asset_id]["assetMetadata"]["boundingBox"][
                    "y"
                ]
                if obj["position"]["y"] + asset_height > wall_height:
                    continue  # if the object is too high, skip it
                placement["position"]["y"] = (
                    obj["position"]["y"] + (asset_height / 2) + 0.001
                )  # add half of the height to the y position and a small offset

                # temporary solution fix position and rotation for thin objects
                if thin:
                    placement = self.fix_placement_for_thin_assets(placement)

                if not small and not thin:
                    placement["kinematic"] = (
                        True  # set kinematic to true for non-small objects
                    )
                    continue  # TODO : remove big assets

                if "breakable" in self.database[asset_id]["objectMetadata"].keys():
                    if self.database[asset_id]["objectMetadata"]["breakable"] == True:
                        placement["kinematic"] = True

                placements.append(placement)

        # TODO: check collision between small objects on the same receptacle
        valid_placements = self.check.check_collision(placements)
        results.extend(valid_placements)

        return results

    def random_select(self, candidates):
        scores = [candidate[1] for candidate in candidates]
        scores_tensor = torch.Tensor(scores)
        probas = F.softmax(
            scores_tensor, dim=0
        )  # TODO: consider using normalized scores
        selected_index = torch.multinomial(probas, 1).item()
        selected_candidate = candidates[selected_index]
        return selected_candidate

    def retrieve_object_by_cat(self, cat):
        candidates = self.object_retriever.retrieve(
            [f"a 3D model of a single {cat}"], self.clip_threshold
        )
        return candidates

    def select_small_objects_per_receptacle(self, large_obj):
        category = large_obj["label"]
        receptacle_dimensions = large_obj["size"]
        receptacle_size = [
            receptacle_dimensions[0] * 100,
            receptacle_dimensions[2] * 100,
        ]
        sorted(receptacle_size)
        receptacle_area = receptacle_size[0] * receptacle_size[1]
        capacity = 0
        num_objects = 0
        results = []

        for small_object in large_obj["objects_on_top"]:
            object_name, quantity, variance_type = (
                small_object["object_name"],
                small_object["quantity"],
                small_object["variance_type"],
            )
            quantity = min(quantity, 5)  # maximum 5 objects per receptacle
            print(f"Selecting {quantity} {object_name} for {category}")
            # Select the object
            candidates = self.object_retriever.retrieve(
                [f"a 3D model of {object_name}"], self.clip_threshold
            )
            candidates = [
                candidate
                for candidate in candidates
                if self.database[candidate[0]]["annotations"]["onObject"] == True
            ]  # Only select objects that can be placed on other objects

            valid_candidates = []  # Only select objects with high confidence

            for candidate in candidates:
                candidate_dimensions = self.database[candidate[0]]["assetMetadata"][
                    "boundingBox"
                ]
                candidate_size = [candidate_dimensions["x"], candidate_dimensions["z"]]
                sorted(candidate_size)
                if (
                    candidate_size[0] < receptacle_size[0] * 0.9
                    and candidate_size[1] < receptacle_size[1] * 0.9
                ):  # if the object is smaller than the receptacle, threshold is 90%
                    valid_candidates.append(candidate)

            if len(valid_candidates) == 0:
                print(f"No valid candidate for {object_name}.")
                continue

            # remove used assets
            top_one_candidate = valid_candidates[0]
            if len(valid_candidates) > 1:
                valid_candidates = [
                    candidate
                    for candidate in valid_candidates
                    if candidate[0] not in self.used_assets
                ]
            if len(valid_candidates) == 0:
                valid_candidates = [top_one_candidate]

            valid_candidates = valid_candidates[:5]  # only select top 5 candidates

            selected_asset_ids = []
            if variance_type == "same":
                selected_candidate = self.random_select(valid_candidates)
                selected_asset_id = selected_candidate[0]
                selected_asset_ids = [selected_asset_id] * quantity

            elif variance_type == "varied":
                for i in range(quantity):
                    selected_candidate = self.random_select(valid_candidates)
                    selected_asset_id = selected_candidate[0]
                    selected_asset_ids.append(selected_asset_id)
                    if len(valid_candidates) > 1:
                        valid_candidates.remove(selected_candidate)

            for i in range(quantity):
                small_object_dimensions = self.database[selected_asset_ids[i]][
                    "assetMetadata"
                ]["boundingBox"]
                small_object_sizes = [
                    small_object_dimensions["x"],
                    small_object_dimensions["y"],
                    small_object_dimensions["z"],
                ]
                sorted(small_object_sizes)
                # small_object_area = small_object_dimensions["x"] * small_object_dimensions["z"]
                # take the maximum 2 dimensions and multiply them
                small_object_area = small_object_sizes[1] * small_object_sizes[2] * 0.8
                capacity += small_object_area
                num_objects += 1
                if capacity > receptacle_area * 0.6 and num_objects > 1:
                    print(f"Warning: {category} is overfilled.")
                    break
                if num_objects > 15:
                    print(f"Warning: {category} has too many objects.")
                    break
                else:
                    results.append((f"{object_name}-{i}", selected_asset_ids[i]))

        ordered_small_objects = []
        for object_name, asset_id in results:
            dimensions = self.database[asset_id]["assetMetadata"]["boundingBox"]
            size = max(dimensions["x"], dimensions["z"])
            ordered_small_objects.append((object_name, asset_id, size))
        ordered_small_objects.sort(key=lambda x: x[2], reverse=True)

        return ordered_small_objects
