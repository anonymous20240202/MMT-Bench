from pathlib import Path
import csv
import json

import mmcv
import numpy as np
from PIL import Image
from iso3166 import countries

from base_dataset import BaseDataset


class tallyqa_complex(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/visual_genome",
        "anno_path": "/path/to/lvlm_evaluation/data/counting/tallyqa/test.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        anno_info_lsit = mmcv.load(self.anno_path)
        for anno_info in anno_info_lsit:
            if anno_info["issimple"]:
                continue
            original_image_path = Path(self.image_path) / anno_info['image']
            answer = anno_info["answer"]
            question = anno_info["question"]

            image_info = self.image_dict
            image_info.update(
                {
                    "question": question,
                    "answer": answer,
                    "original_image_path": str(original_image_path)
                }
            )
            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info):
        def get_answer(gt, rangee=4, num=3):
            import random
            a = list(range(max(0, gt - rangee), gt + rangee))
            a.remove(gt)
            return random.sample(a, k=num)
        
        question = image_info["question"]

        qa_json = {
            "question": question,
            "num_wrong_choices": 3,
            "wrong_choices_list": get_answer(image_info["answer"]),
            "gt": image_info["answer"]
        }

        while True:
            try:
                qa_json = BaseDataset.post_process(qa_json)
                break
            except:
                pass

        return qa_json
