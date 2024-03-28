import os
from pathlib import Path
from collections import OrderedDict

from base_dataset import BaseDataset

import mmcv

import math

from tqdm import tqdm

import math

def rotate_box_to_xyxy(box):
    cx, cy, w, h, angle = float(box['cx']), float(box['cy']), float(box['w']), float(box['h']), float(box['angle'])
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Calculate the coordinates of the four corners of the rotated bounding box
    theta = -angle * math.pi / 180
    x_corners = [x1, x2, x2, x1]
    y_corners = [y1, y1, y2, y2]
    x_corners_rot = []
    y_corners_rot = []
    for i in range(4):
        x_corners_rot.append(cx + (x_corners[i] - cx) * math.cos(theta) - (y_corners[i] - cy) * math.sin(theta))
        y_corners_rot.append(cy + (x_corners[i] - cx) * math.sin(theta) + (y_corners[i] - cy) * math.cos(theta))

    return min(x_corners_rot), min(y_corners_rot), max(x_corners_rot), max(y_corners_rot)


from PIL import Image

def crop_and_save_image_center(image_path, coordinates, save_path):
    with Image.open(image_path) as img:
        crop_box = coordinates

        # Crop and save the image
        cropped_img = img.crop(crop_box)
        cropped_img.save(save_path)

class OWAD(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/taskonomy_data/weapon_recognition/OWAD",
        "anno_path": "/path/to/taskonomy_data/weapon_recognition/OWAD/meta_info.json",
        "save_image_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/weapon_recognition/images",
        "sampling_num": 60,
        "url": "internet",
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)

        for image_info in tqdm(anno_info["images"]):
            image_info["original_image_path"] = os.path.join(self.image_path, image_info["image_name"])

            for box, label in zip(image_info["bbox_or_rbbox"], image_info["image_label"]):
                if "xmin" in box.keys():
                    crop_box = [int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])]
                else:
                    crop_box = rotate_box_to_xyxy(box)
                new_save_image = os.path.join(self.save_image_path, self.new_image_name())

                crop_and_save_image_center(image_info["original_image_path"], crop_box, new_save_image)

                image_info["original_image_path"] = new_save_image
                image_info["category"] = label
                image_info["source"] = self.dataset_name

                self.images_info.append(image_info)


        self.dataset_info["category_space"] = ["gun", "pistol"]
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        question = "What is the category of weapon in the picture?"

        num_choices = 2
        if image_info["category"] == "gun":
            wrong_choices_list = ["pistol"]
        elif image_info["category"] == "pistol":
            wrong_choices_list = ["gun"]
        else:
            raise NotImplementedError

        qa_json = {
            "num_wrong_choices": num_choices - 1,
            "gt": image_info["category"],
            "question": question,
            "wrong_choices_list": wrong_choices_list
        }

        qa_json = BaseDataset.post_process(qa_json, question=question)

        return qa_json


class weapon_detection_dataset(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/taskonomy_data/weapon_recognition/weapon_detection_dataset",
        "anno_path": "/path/to/taskonomy_data/weapon_recognition/weapon_detection_dataset/meta_info.json",
        "save_image_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/weapon_recognition/images",
        "sampling_num": 140,
        "url": "internet",
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):

        def crop_and_save_image_center(image_path, coordinates, save_path):
            """
            Crops an image based on given center coordinates and dimensions (cxcywh) and saves the cropped image to a specified path.
            The coordinates are in the format (cx, cy, w, h) where each value is a floating-point number between 0 and 1.

            :param image_path: Path to the original image.
            :param coordinates: A tuple (cx, cy, w, h) of the crop box center and size.
            :param save_path: Path where the cropped image will be saved.
            """
            # Open the image
            with Image.open(image_path) as img:
                width, height = img.size

                # Convert fractional coordinates to absolute pixel coordinates
                cx, cy, w, h = coordinates
                x = int((cx - w / 2) * width)
                y = int((cy - h / 2) * height)
                w = int(w * width)
                h = int(h * height)

                # Define the crop box
                crop_box = (x, y, x + w, y + h)

                # Crop and save the image
                img = img.convert("RGB")
                cropped_img = img.crop(crop_box)
                cropped_img.save(save_path, "JPEG")

        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)

        for image_info in tqdm(anno_info["images"]):
            image_info["original_image_path"] = os.path.join(self.image_path, image_info["image_name"])

            box = image_info["bbox"]
            new_save_image = os.path.join(self.save_image_path, self.new_image_name())
            crop_and_save_image_center(image_info["original_image_path"], box, new_save_image)

            image_info["original_image_path"] = new_save_image
            image_info["category"] = image_info["image_label"]
            image_info["source"] = self.dataset_name

            self.images_info.append(image_info)


        self.dataset_info["category_space"] = list(anno_info["category_space"].keys())
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the category of weapon in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Grenade Launcher",
                "question": question,
                "wrong_choices_list": ["Handgun", "Knife", "SMG"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["category"],
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)

        i = 0
        while i <= 10:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json

