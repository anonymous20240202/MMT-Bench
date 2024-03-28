from pathlib import Path
import csv
import os
import mmcv
from PIL import Image
import struct
import numpy as np

from base_dataset import BaseDataset
from prompt.utils import *


class CNNGestureRecognizer(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/visual_recognition/gesture_recognition/CNNGestureRecognizer/imgfolder_b",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.category_space = [
            "OK", "PEACE", "STOP", "PUNCH"#, "NOTHING"
        ]
        self.dataset_info["category_space"] = self.category_space
    
    def parse_images_info(self):
        self.images_info = list()

        im_list = os.listdir(self.DATA_METAINFO['image_path'])
        if '.ipynb_checkpoints' in im_list:
            im_list.remove('.ipynb_checkpoints')
        for im in im_list:
            original_image_path = os.path.join(self.DATA_METAINFO['image_path'], im)
            if 'iiok' in im:# and 'iii' not in im:
                cate = "OK"
            elif 'peace' in im:
                cate = "PEACE"
            elif 'punch' in im:
                cate = "PUNCH"
            elif 'stop' in im:
                cate = "STOP"
            else:
                continue
        
            info = {
                'source': self.dataset_name,
                'category': cate,
                'original_image_path': original_image_path.replace('/cpfs01/user/linyuqi/datasets', ' /path/to'),
                }
            self.images_info.append(info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What category of the gesture is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": [replace_underscore_with_space(i) for i in dataset_info["category_space"]],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "OK",
                "question": question,
                "wrong_choices_list": ["PEACE", "STOP", "PUNCH"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": replace_underscore_with_space(image_info["category"]),
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


class gesture_digits(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/visual_recognition/gesture_recognition/gesture_digits/picture",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.category_space = [
            "zero", "one", "two", "three", "four",  "five",  "six",  "seven",  "eight",  "nine",  "ten", 
        ]
        self.dataset_info["category_space"] = self.category_space
        self.cate_map = {}
        for idx, cate in enumerate(self.category_space):
            self.cate_map[idx] = cate
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        
        im_list = os.listdir(self.DATA_METAINFO['image_path'])
        if '.ipynb_checkpoints' in im_list:
            im_list.remove('.ipynb_checkpoints')
            
        for im in im_list:
            if len(im.split('_')) == 2:
                cateid, _ = im.split('_')
            else:
                cateid = im.replace('.jpg', '')
            original_image_path = os.path.join(self.DATA_METAINFO['image_path'], im)
            
            info = {
                'source': self.dataset_name,
                'category': self.cate_map[int(cateid)],
                'original_image_path': original_image_path.replace('/cpfs01/user/linyuqi/datasets', ' /path/to'),
                }
            self.images_info.append(info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What category of the gesture is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": [replace_underscore_with_space(i) for i in dataset_info["category_space"]],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "zero",
                "question": question,
                "wrong_choices_list": ["one", "two", "four"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": replace_underscore_with_space(image_info["category"]),
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