from pathlib import Path

import struct
import numpy as np

import mmcv

from base_dataset import BaseDataset


class HAR(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/action_recognition/image_based_action_recognition/metadata_info.json",
        "sampling_num": 100,
        "visual_input_component": ["natural_image", ],
        "dataset_description": "This dataset comes from kaggle: https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset",
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)
        for image_info in anno_info["images"]:

            if image_info["source"] != "HAR":
                continue

            image_info["source"] = self.dataset_name
            self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = anno_info["dataset_list"][0]["category_space"]
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv
        import random

        num_choices = 4
        question = "What is the action type shown in this image?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "sitting",
                "question": question,
                "wrong_choices_list": ["clapping", "dancing", "laughing"]
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


class POLAR(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/action_recognition/image_based_action_recognition/metadata_info.json",
        "sampling_num": 100,
        "visual_input_component": ["natural_image", ],
        "dataset_description": "This dataset comes from here: https://data.mendeley.com/datasets/hvnsh7rwz7/1",
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)
        for image_info in anno_info["images"]:

            if image_info["source"] != "POLAR":
                continue
            
            if len(image_info["category"]) > 1:
                continue

            image_info["source"] = self.dataset_name
            image_info["category"] = image_info["category"][0]
            self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = anno_info["dataset_list"][1]["category_space"]
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv
        import random

        num_choices = 4
        question = "What is the action type shown in this image?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "bendover",
                "question": question,
                "wrong_choices_list": ["squat", "run", "jump"]
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
    