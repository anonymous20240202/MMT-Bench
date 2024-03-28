import os
import json
import sys
import csv
import numpy as np
from PIL import Image
sys.path.append("data_process")
from pathlib import Path

import mmcv
from shapely.geometry import Polygon

from base_dataset import BaseDataset

from prompt.utils import *


class RefCOCO_refer(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/coco/train2014",
        "anno_path": "/path/to/Referring_Detection/refer/data/metadata.json",
        "sampling_num": 50,
        "visual_input_component": ["natural_image",],
        "dataset_description": "This referring expression generation (REG) dataset was collected using the ReferitGame. In this two-player game, the first player is shown an image with a segmented target object and asked to write a natural language expression referring to the target object. The second player is shown only the image and the referring expression and asked to click on the corresponding object. If the players do their job correctly, they receive points and swap roles. If not, they are presented with a new object and image for description. Images in these collections were selected to contain two or more objects of the same object category. In the RefCOCO dataset, no restrictions are placed on the type of language used in the referring expressions. In a version of this dataset called RefCOCO+ players are disallowed from using location words in their referring expressions by adding “taboo” words to the ReferItGame. This dataset was collected to obtain a referring expression dataset focsed on purely appearance based description, e.g., “the man in the yellow polka-dotted shirt” rather than “the second man from the left”, which tend to be more interesting from a computer vision based perspective and are independent of viewer perspective. RefCOCO consists of 142,209 refer expressions for 50,000 objects in 19,994 images, and RefCOCO+ has 141,564 expressions for 49,856 objects in 19,992 images."
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)
        for image_info in anno_info:
            image_info["source"] = self.dataset_name
            image_info["original_image_path"] = os.path.join(self.image_path, image_info["original_image_path"])

            self.images_info.append(image_info)
    

    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import mmcv
        import numpy as np
        import json
        import random

        num_choices = 4

        width, height = image_info["width"], image_info["height"]
        question = random.choice(image_info["sentence"])
        question = f"Please provide the bounding box coordinates for the following description of the object of interest. Provide the output for the detected area in the format [x, y, w, h]. This format represents the bounding box, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as its width and height. Note that the width of the input image is {width} and the height is {height}.\nQUESTION:{question}"
        bbox = process_bbox(image_info["bounding_box_coordinates"], width=width, height=height, i="fxyxy", o="fxywh")

        # question = f""

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_bounding_box_from_single_bbox(bbox, width, height, num_choices-1)
                    
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": bbox,
                    "question": question,
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json


class RefCOCOplus_refer(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/coco/train2014",
        "anno_path": "/path/to/Referring_Detection/refer/data/metadata+.json",
        "sampling_num": 50,
        "visual_input_component": ["natural_image",],
        "dataset_description": "This referring expression generation (REG) dataset was collected using the ReferitGame. In this two-player game, the first player is shown an image with a segmented target object and asked to write a natural language expression referring to the target object. The second player is shown only the image and the referring expression and asked to click on the corresponding object. If the players do their job correctly, they receive points and swap roles. If not, they are presented with a new object and image for description. Images in these collections were selected to contain two or more objects of the same object category. In the RefCOCO dataset, no restrictions are placed on the type of language used in the referring expressions. In a version of this dataset called RefCOCO+ players are disallowed from using location words in their referring expressions by adding “taboo” words to the ReferItGame. This dataset was collected to obtain a referring expression dataset focsed on purely appearance based description, e.g., “the man in the yellow polka-dotted shirt” rather than “the second man from the left”, which tend to be more interesting from a computer vision based perspective and are independent of viewer perspective. RefCOCO consists of 142,209 refer expressions for 50,000 objects in 19,994 images, and RefCOCO+ has 141,564 expressions for 49,856 objects in 19,992 images."
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)
        for image_info in anno_info:
            image_info["source"] = self.dataset_name
            image_info["original_image_path"] = os.path.join(self.image_path, image_info["original_image_path"])

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import mmcv
        import numpy as np
        import json
        import random

        num_choices = 4

        width, height = image_info["width"], image_info["height"]
        question = random.choice(image_info["sentence"])
        question = f"Please provide the bounding box coordinates for the following description of the object of interest. Provide the output for the detected area in the format [x, y, w, h]. This format represents the bounding box, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as its width and height. Note that the width of the input image is {width} and the height is {height}.\nQUESTION:{question}"
        bbox = process_bbox(image_info["bounding_box_coordinates"], width=width, height=height, i="fxyxy", o="fxywh")

        # question = f""

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_bounding_box_from_single_bbox(bbox, width, height, num_choices-1)
                    
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": bbox,
                    "question": question,
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json


class RefCOCOg_refer(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/coco/train2014",
        "anno_path": "/path/to/Referring_Detection/refer/data/metadatag.json",
        "sampling_num": 100,
        "visual_input_component": ["natural_image",],
        "dataset_description": "gRefCOCO is the first large-scale Generalized Referring Expression Segmentation dataset that contains multi-target, no-target, and single-target expressions."
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)
        for image_info in anno_info:
            image_info["source"] = self.dataset_name
            image_info["original_image_path"] = os.path.join(self.image_path, image_info["original_image_path"])

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import mmcv
        import numpy as np
        import json
        import random

        num_choices = 4

        width, height = image_info["width"], image_info["height"]
        question = random.choice(image_info["sentence"])
        question = f"Please provide the bounding box coordinates for the following description of the object of interest. Provide the output for the detected area in the format [x, y, w, h]. This format represents the bounding box, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as its width and height. Note that the width of the input image is {width} and the height is {height}.\nQUESTION:{question}"
        bbox = process_bbox(image_info["bounding_box_coordinates"], width=width, height=height, i="fxyxy", o="fxywh")

        # question = f""

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_bounding_box_from_single_bbox(bbox, width, height, num_choices-1)
                    
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": bbox,
                    "question": question,
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json
