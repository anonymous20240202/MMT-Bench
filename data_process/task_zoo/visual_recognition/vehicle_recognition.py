from pathlib import Path
import csv
import os
import mmcv
from PIL import Image
import struct
import numpy as np

from base_dataset import BaseDataset

from prompt.utils import *


class TAU_Vehicle_Type_Recognition(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/visual_recognition/vehicle_recognition/TAU_Vehicle_Type_Recognition/train/train",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "only train set have annotations"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        
        cates = os.listdir(self.DATA_METAINFO['image_path'])
        if '.ipynb_checkpoints' in cates:
            cates.remove('.ipynb_checkpoints')
        
        for cate in cates:
            im_dir = os.path.join(self.DATA_METAINFO['image_path'], cate)
            im_list = os.listdir(im_dir)
            cate = cate.replace("_", " ")
            self.category_space.append(replace_underscore_with_space(cate))
            
            for im in im_list:
                original_image_path = os.path.join(im_dir, im)
                info = {
                    'source': self.dataset_name,
                    'category': replace_underscore_with_space(cate),
                    'original_image_path': original_image_path.replace('/cpfs01/user/linyuqi/datasets', ' /path/to'),
                    }
                self.images_info.append(info)
        self.dataset_info["category_space"] = self.category_space

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What category of the vehicle is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": [replace_underscore_with_space(i) for i in dataset_info["category_space"]],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "segway",
                "question": question,
                "wrong_choices_list": ["cart", "barge", "truck"]
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

class vehicle_type_recognition(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/visual_recognition/vehicle_recognition/vehicle-type-recognition",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "no val and test set."
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        

    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        
        cates = os.listdir(self.DATA_METAINFO['image_path'])
        if '.ipynb_checkpoints' in cates:
            cates.remove('.ipynb_checkpoints')
        
        for cate in cates:
            
            im_dir = os.path.join(self.DATA_METAINFO['image_path'], cate)
            im_list = os.listdir(im_dir)
            cate = cate.replace("_", " ")
            self.category_space.append(replace_underscore_with_space(cate))
            
            for im in im_list:
                original_image_path = os.path.join(im_dir, im)
                info = {
                    'source': self.dataset_name,
                    'category': replace_underscore_with_space(cate),
                    'original_image_path': original_image_path.replace('/cpfs01/user/linyuqi/datasets', ' /path/to'),
                    }
                self.images_info.append(info)
        self.dataset_info["category_space"] = self.category_space

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What category of the vehicle is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": [replace_underscore_with_space(i) for i in dataset_info["category_space"]],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "truck",
                "question": question,
                "wrong_choices_list": ["bus", "car", "motorcycle"]
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
