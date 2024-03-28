from pathlib import Path

import struct
import numpy as np

import mmcv

from base_dataset import BaseDataset


class adobe_vfr(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/samples/font_classification/metadata_info_new.json",
        "sampling_num": 200,
        "visual_input_component": ["text-rich_image", ],
        "dataset_description": "AdobeVFR Dataset\u807dA large set of\u807dlabeled real-world\u807dimages as well as a large corpus of unlabeled real-world data are collected for both training and testing, which is the first of its kind",
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)
        for image_info in anno_info["image"]:
            
            image_info["category"] = image_info["label"]
            self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = anno_info["dataset_list"][0]["category_space"]
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv
        import random

        num_choices = 4
        question = "What is the font in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": random.sample(dataset_info["category_space"], 500),
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "UniversLTStd-XBlack",
                "question": question,
                "wrong_choices_list": ["UtopiaStd-Bold", "UtopiaStd-CaptIt", "UtopiaStd-Regular"]
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
    