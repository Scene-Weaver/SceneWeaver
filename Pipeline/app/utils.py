import base64
import json
import math
import random
from typing import Any, Dict

# def extract_json(input_string):
#     # Using regex to identify the JSON structure in the string
#     json_match = re.search(r"{.*}", input_string, re.DOTALL)
#     if json_match:
#         extracted_json = json_match.group(0)
#         try:
#             # Convert the extracted JSON string into a Python dictionary
#             json_dict = json.loads(extracted_json)
#             json_dict = check_dict(json_dict)
#             return json_dict
#         except json.JSONDecodeError:
#             print(input_string)
#             print("Error while decoding the JSON.")
#             return None
#     else:
#         print("No valid JSON found.")
#         return None


def extract_json(input_string):
    # Step 1: Extract the JSON string
    start_idx = None
    brace_count = 0
    json_string = ""
    for i, char in enumerate(input_string):
        if char == "{":
            if start_idx is None:
                start_idx = i  # Mark the start of the JSON
            brace_count += 1
        elif char == "}":
            brace_count -= 1
        if start_idx is not None:
            json_string += char
        if brace_count == 0 and start_idx is not None:
            # Found the complete JSON structure
            break
    if not json_string:
        raise ValueError("No valid JSON found in the input string.")
    # Step 2: Convert the JSON string to a dictionary
    try:
        json_dict = json.loads(json_string)
        return json_dict
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON: {e}")


def check_dict(dict):
    valid = True
    attributes = ["index", "category", "size", "objects_on_top", "objects_inside"]
    for key, value in dict.items():
        if not isinstance(key, str):
            valid = False
            break

        if not isinstance(value, Dict):
            valid = False
            break

        for attribute in attributes:
            if attribute not in value:
                valid = False
                break

        if not isinstance(value["index"], int):
            valid = False
            break

        if not isinstance(value["category"], str):
            valid = False
            break

        if (
            not isinstance(value["size"], list)
            or len(value["size"]) != 3
            or not all(isinstance(i, int) for i in value["size"])
        ):
            dict[key]["size"] = None

        if not isinstance(value["objects_on_top"], list):
            dict[key]["objects_on_top"] = []

        if not isinstance(value["objects_inside"], list):
            dict[key]["objects_inside"] = []

        for name in ["objects_on_top", "objects_inside"]:
            for i, child in enumerate(value[name]):
                if not isinstance(child, Dict):
                    valid = False
                    break

                for attribute in ["object_name", "quantity", "variance_type"]:
                    if attribute not in child:
                        valid = False
                        break

                if not isinstance(child["object_name"], str):
                    valid = False
                    break

                if not isinstance(child["quantity"], int):
                    dict[key][name][i]["quantity"] = 1

                if not isinstance(child["variance_type"], str) or child[
                    "variance_type"
                ] not in ["same", "varied"]:
                    dict[key][name][i]["variance_type"] = "same"

    if not valid:
        return None
    else:
        return dict


def custom_distribution():
    while True:
        sample = random.uniform(0, 1)
        probability = math.exp(-((sample - 0.5) ** 2) / 0.02)
        if random.uniform(0, 1) < probability:
            return sample


def get_asset_metadata(obj_data: Dict[str, Any]):
    if "assetMetadata" in obj_data:
        return obj_data["assetMetadata"]
    elif "thor_metadata" in obj_data:
        return obj_data["thor_metadata"]["assetMetadata"]
    else:
        raise ValueError("Can not find assetMetadata in obj_data")


def get_bbox_dims(obj_data: Dict[str, Any]):
    am = get_asset_metadata(obj_data)

    bbox_info = am["boundingBox"]

    if "x" in bbox_info:
        return bbox_info

    if "size" in bbox_info:
        return bbox_info["size"]

    mins = bbox_info["min"]
    maxs = bbox_info["max"]

    return {k: maxs[k] - mins[k] for k in ["x", "y", "z"]}


def dict2str(d, indent=0):
    """
    Convert a dictionary into a formatted string.

    Parameters:
    - d: dict, the dictionary to convert.
    - indent: int, the current indentation level (used for nested structures).

    Returns:
    - str: The string representation of the dictionary.
    """
    if not isinstance(d, dict):
        raise ValueError("Input must be a dictionary")

    result = []
    indent_str = " " * (indent * 4)  # Indentation for nested levels

    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            result.append(
                f"{indent_str}{key}: {{\n{dict2str(value, indent + 1)}\n{indent_str}}}"
            )
        elif isinstance(value, list):
            # Handle lists
            # list_str = ", ".join(
            #     dict2str(item, indent + 1) if isinstance(item, dict) else str(item)
            #     for item in value
            # )
            list_str = ", ".join(
                dict2str(item, indent + 1)
                if isinstance(item, dict)
                else f"{item:.2f}"
                if isinstance(item, float)
                else str(item)
                for item in value
            )
            result.append(f"{indent_str}{key}: [{list_str}]")
        else:
            # Handle other types
            result.append(f"{indent_str}{key}: {repr(value)}")

    return "{" + ",\n".join(result) + "}"


def lst2str(lst):
    if isinstance(lst[0], list):
        s = ["[" + ", ".join(list(map(str, i))) + "]" for i in lst]
        s = "\n".join(s)
        return s
    else:
        lst = list(map(str, lst))
        return "[" + ", ".join(lst) + "]"


def encode_image(image_path):
    """
    Encodes image located at @image_path so that it can be included as part of GPT prompts

    Args:
        image_path (str): Absolute path to image to encode

    Returns:
        str: Encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


if __name__ == "__main__":
    s = '{\n  "iter": 0,\n  "Analysis of current scene": "The current scene is completely empty with no layout or objects present. This is the initial stage of scene creation, and the primary goal is to establish a foundational layout that can later be enhanced with details and objects.",\n  "Thoughts": "Given that the scene is starting from scratch, the first step should be to create a basic layout. Method 1 (real2sim indoor scene data) and Method 2 (scene synthesis by neural network) are both suitable for this task. However, Method 1 is based on real-world data, providing a more accurate and realistic layout, which is crucial for establishing a solid foundation for the scene.",\n  "Recommendation": "For this iteration, I recommend using Method 1 (real2sim indoor scene data) to generate the initial layout. This method will provide a realistic and accurate foundation based on real indoor scenes, which is essential for building a believable 3D environment. Subsequent iterations can focus on adding details, objects, and further customization using other methods like Method 3 (image generation + 3D reconstruction) or Method 4 (Generated Scene Using GPT).",\n  "Method number": 1,\n  "Ideas": "Generate foundational layout for a common room type such as a living room or bedroom, depending on the user\'s specific needs.",\n  "RoomType": "Specify the room type based on user\'s requirement (e.g., living room, bedroom)"\n}'

    print(extract_json(s))
