import os
import json
import sys
import csv
import numpy as np
from PIL import Image
from pathlib import Path

import mmcv
from tqdm import tqdm

from base_dataset import BaseDataset


class artemis(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/lvlm_evaluation/Artwork_Emotion_Recognition/Artwork_Emotion_Recognition.json",
        "sampling_num": 200,
        "dataset_description": "xxx",
        "visual_input_component": ["natural_image", ]
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = []
        
        anno_info = mmcv.load(self.anno_path)
        self.category_space = []

        for image_info in tqdm(anno_info["images"]):

            if not os.path.exists(image_info["original_image_path"]):
                continue
            self.category_space.append(image_info["emotion"])
            image_info["source"] = "artemis"
            self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = list(set(self.category_space))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What emotion is expressed in the artwork in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "amusement",
                "question": question,
                "wrong_choices_list": ["disgust", "anger", "awe"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["emotion"],
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
