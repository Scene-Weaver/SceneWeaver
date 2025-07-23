import base64

from TongGPT import GPT4o


class GPT4(GPT4o):
    """
    Simple interface for interacting with GPT-4O model
    """

    VERSIONS = {
        "4v": "gpt-4-vision-preview",
        "4o": "gpt-4o-2024-08-06",
        "4o-mini": "gpt-4o-mini",
        "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
    }

    def __init__(
        self,
        api_key=None,
        version="gpt-4-turbo",
    ):
        # def __init__(self, '
        self.version = version
        MODEL = self.VERSIONS[version]
        self.MODEL = MODEL
        REGION = "eastus2"
        super().__init__(MODEL, REGION)

    def __call__(self, payload, verbose=False):
        """
        Queries GPT using the desired @prompt

        Args:
            payload (dict): Prompt payload to pass to GPT. This should be formatted properly, see
                https://platform.openai.com/docs/overview for details
            verbose (bool): Whether to be verbose as GPT is being queried

        Returns:
            None or str: Raw outputted GPT response if valid, else None
        """
        if verbose:
            print(f"Querying GPT-{self.version} API...")
        # import pdb
        # pdb.set_trace()
        response = self.send_request(payload)
        try:
            content = response.choices[0].message.content
        except:
            print(
                f"Got error while querying GPT-{self.version} API! Response:\n\n{response}"
            )
            return None

        if verbose:
            print(f"Finished querying GPT-{self.version}.")

        return content

    def encode_image(self, image_path):
        """
        Encodes image located at @image_path so that it can be included as part of GPT prompts

        Args:
            image_path (str): Absolute path to image to encode

        Returns:
            str: Encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_payload(self, prompting_text_system, prompting_text_user):
        text_dict_system = {"type": "text", "text": prompting_text_system}
        content_system = [text_dict_system]

        content_user = [{"type": "text", "text": prompting_text_user}]

        object_caption_payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": content_system},
                {"role": "user", "content": content_user},
            ],
            "temperature": 0,
            "max_tokens": 4096,
        }
        return object_caption_payload

    def get_payload_scene_image(
        self, prompting_text_system, prompting_text_user, render_path=None
    ):
        text_dict_system = {"type": "text", "text": prompting_text_system}
        content_system = [text_dict_system]

        if render_path is not None:
            imgs_base64 = self.encode_image(render_path)
            img_dict = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{imgs_base64}"},
            }

            prompting_user_lst = prompting_text_user.split("SCENE_IMAGE")
            content_user = [
                {"type": "text", "text": prompting_user_lst[0]},
                img_dict,
                {"type": "text", "text": prompting_user_lst[1]},
            ]
        else:
            content_user = [
                {
                    "type": "text",
                    "text": prompting_text_user.replace("SCENE_IMAGE", "None"),
                }
            ]

        object_caption_payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": content_system},
                {"role": "user", "content": content_user},
            ],
            "temperature": 0,
            "max_tokens": 4096,
        }
        return object_caption_payload

    def payload_front_pose(
        self,
        category,
        candidates_fpaths,
    ):
        """
        Given a list of candidate snapshots, return the payload used to find the nearest neighbor in terms of orientation.

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        cand_imgs_base64 = [self.encode_image(img) for img in candidates_fpaths]

        prompt_text_system = (
            f"You have been given 2 or 4 images of an {category}, each taken from a different angle. Your task is to identify the image that shows the 'front view' of the object. The front view refers to the perspective where the object's main face or most important features are most clearly visible, typically from the viewer’s point of view.\n\n"
            + "Please keep the following in mind:\n"
            + "1. The front view is often characterized by the most significant or most visible face of the object.\n"
            + "2. For objects like cabinets, the front view is typically where the doors and drawers are visible. For chairs, the front view may show the seat and backrest. For other objects, consider the main or most notable side visible from the viewer's point of view.\n"
            + "3. The front view is usually the view where the object faces the camera directly or is oriented in such a way that the most prominent features (such as a face, label, or handle) are visible.\n"
            + "4. If the images are taken at different angles (front, right, left, back), choose the image where the object faces you (from the viewer’s perspective).\n"
            + "5. Given these considerations, please only return the index number of the image that represents the 'front view'. The indices of the images start from 0.\n"
            + "6. Retun only the index number without any other words.\n"
            + "Example output:1"
        )
        content_system = [{"type": "text", "text": prompt_text_system}]

        content = [
            {
                "type": "text",
                "text": "The following images show 4 orientations of the asset with starting index 0:\n\n",
            }
        ]

        for i in range(len(cand_imgs_base64)):
            text_prompt = f"orientation {i}:\n"
            text_dict = {"type": "text", "text": text_prompt}
            img_dict = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{cand_imgs_base64[i]}"},
            }
            content.append(text_dict)
            content.append(img_dict)

        content.append(
            {
                "type": "text",
                "text": "Please take a moment to relax and carefully select the orientation that most closely matches the target object from the viewer's point of view, without providing an explanation. Kindly follow all instructions precisely.\n",
            }
        )

        NN_payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": content_system},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "max_tokens": 50,
        }
        return NN_payload

    def payload_roomtype(self, obj_cnts):
        """
        Given a list of candidate snapshots, return the payload used to find the nearest neighbor in terms of
        orientation.

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string

        # prompt_text_system = f"You are given a description of objects in a room.\n"+\
        #                       "Based on the list of objects and their quantities, predict the room type and the confidence score between 0 and 1, where 1 means a very high confidence in the prediction.\n"+\
        #                       "The description includes the object categories and their counts.\n"+\
        #                       "The room type can be things like 'living room,' 'kitchen,' 'bedroom,' 'office,' etc.\n"+\
        #                       "Please provide the predicted room type and confidence score based on the objects provided.\n"+\
        #                       "Example Input: 'The room has 1 sofa, 1 coffee table, and 2 chairs.\n"+\
        #                       "Predict the room type and confidence score based on the provided description.\n"+\
        #                       "Retun only the room type and confidence score without any other words. Separate with a comma.\n"+\
        #                       "Example output: bedroom, 0.9"

        prompt_text_system = (
            "You are given a description of objects in a room or multiple rooms.\n"
            + "Based on the list of objects and their quantities, predict the room type(s) and the confidence score(s) between 0 and 1, where 1 means very high confidence in the prediction.\n"
            + "The description includes the object categories and their counts.\n"
            + "The room type(s) can be things like 'living room,' 'kitchen,' 'bedroom,' 'office,' etc.\n"
            + "Please provide the predicted room type(s) and confidence score(s), separated by a comma. If there are multiple room types, separate them with a semicolon. The confidence score(s) should correspond to each room type.\n"
            + "Example Input: 'The room has 1 sofa, 1 coffee table, and 2 chairs.\n"
            + "Predict the room type and confidence score based on the provided description.\n"
            + "Retun only the room type and confidence score without any other words. Separate with a comma.\n"
            + "Example output for single room type: bedroom, 0.9 \n"
            + "Example output for multiple room types: living room, 0.8; office, 0.7"
        )

        content_system = [{"type": "text", "text": prompt_text_system}]

        content = [
            {
                "type": "text",
                "text": f"The room has {obj_cnts}. Kindly follow all instructions precisely and tell me what is the roomtype? ",
            }
        ]

        NN_payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": content_system},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "max_tokens": 50,
        }
        return NN_payload

    def payload_simplify_cnts(self, obj_info):
        """
        Given a list of candidate snapshots, return the payload used to find the nearest neighbor in terms of
        orientation.

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string

        prompt_text_system = (
            "You are given a json of objects with their quantities in a room. Your task is to:\n"
            + "1. Merge similar categories (e.g., 'books' and 'book' should be merged into 'books').\n"
            + "2. Simplify category names (e.g., 'a blue bottle' should become 'bottle' and 'a white hotel information booklet' should be 'booklet').\n"
            + "3. Convert plural forms to singular (e.g., 'books' should be converted to 'book').\n"
            + "After making these changes, return the updated json with merged categories, simplified names, and singular forms.\n\n"
            + "Example Input: {'books': 21, 'book': 5, 'blinds': 1, 'office chair': 9, 'picture': 1, 'table': 2, 'radiator': 1, 'tray': 1, 'a red cup with pens': 1, 'bookshelf': 1, 'case': 1, 'board': 1, 'plant': 5, 'trash can': 1, 'window': 1, 'a blue bottle': 1, 'cabinet': 3, 'box': 1}\n\n"
            + "Expected Output: {'book': 26, 'blind': 1, 'office chair': 9, 'picture': 1, 'table': 2, 'radiator': 1, 'tray': 1, 'cup with pens': 1, 'bookshelf': 1, 'case': 1, 'board': 1, 'plant': 5, 'trash can': 1, 'window': 1, 'bottle': 1, 'cabinet': 3, 'box': 1}+\n"
        )

        content_system = [{"type": "text", "text": prompt_text_system}]

        content = [{"type": "text", "text": f"Input: {obj_info}"}]
        content.append(
            {
                "type": "text",
                "text": "Please take a moment to relax and carefully return the json-format output without providing an explanation. Kindly follow all instructions precisely.\n",
            }
        )

        NN_payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": content_system},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "max_tokens": 1000,
        }
        return NN_payload
