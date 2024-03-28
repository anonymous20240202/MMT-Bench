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


class reason_seg(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/ReasonSeg/val",
        "image_save_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_grounding/reason_seg",
        "sampling_num": 200,
        "visual_input_component": ["natural_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()
        for image_path in Path(self.image_path).iterdir():
            # image_info = dict()
            if image_path.suffix == ".json":
                continue
            try:
                anno_json = image_path.with_suffix(".json")
                anno_info = mmcv.load(anno_json)
                question = anno_info["text"][0]
                polygon_coords = anno_info["shapes"][0]["points"]
            except:
                continue

            polygon = Polygon(polygon_coords)
            # 获取多边形的最小包围框的左上角和右下角坐标
            x1, y1, x2, y2 = polygon.bounds

            bounding_box_coordinates = [round(x1), round(y1), round(x2), round(y2)]

            # img = Image.open(image_path)
            # # 获取图片的大小（宽度和高度）
            # width, height = img.size
            # bounding_box_coordinates = [x1, y1, x2, y2]

            image_info = self.image_dict("")
            image_info["bounding_box_coordinates"] = bounding_box_coordinates
            image_info["question"] = question
            image_info["original_image_path"] = str(image_path)

            self.images_info.append(image_info)
    

    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import mmcv
        import numpy as np
        import json
        import random

        num_choices = 4

        width, height = image_info["width"], image_info["height"]
        question = f"Please provide the bounding box coordinates for the following description of the object of interest. Provide the output for the detected area in the format [x, y, w, h]. This format represents the bounding box, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as its width and height. Note that the width of the input image is {width} and the height is {height}.\nQUESTION:{image_info['question']}"
        bbox = image_info["bounding_box_coordinates"]

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
        
        
        
        
        # sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/general_prompt.txt'))

        # gt = image_info["text"]

        # if len(gt) >= 1000:
        #     return None
        # else:
        #     num_choices = 2
        # input_json = {
        #     "question": question,
        #     "example_dict": {
        #         "num_wrong_choices": num_choices - 1,
        #         "gt": "<html><body><table><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table></body></html>",
        #         "question": "Please read the table in this image and return a html-style reconstructed table in text, do not omit anything.",
        #         "wrong_choices_list": ["<html><body><table><tr><td></td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table></body></html>", "22/01/2018", "22/06/2017"]
        #     },
        #     "query_dict": {
        #         "num_wrong_choices": num_choices - 1,
        #         "gt": gt,
        #         "question": question,
        #     }
        # }

        # user_prompt = json.dumps(input_json)

        # i = 0
        # while i <= 10:
        #     try:
        #         qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
        #         qa_json = BaseDataset.post_process(qa_json, question=question)
        #         break
        #     except:
        #         i += 1

        # return qa_json
