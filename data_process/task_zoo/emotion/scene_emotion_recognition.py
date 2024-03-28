from pathlib import Path

import struct
import numpy as np

import mmcv

from base_dataset import BaseDataset


class Artphoto(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/emotional_quotient_test/scene_emotion/testImages_artphoto",
        "sampling_num": 200,
        "visual_input_component": ["natural_image", ],
        "dataset_description": "xxx",
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        # anno_info = mmcv.load(self.anno_path)
        category_space = []
        for image_path in Path(self.image_path).iterdir():

            category = image_path.stem.split("_")[0]

            image_info = self.image_dict("")
            image_info.update(
                {
                    "category": category,
                    "original_image_path": str(image_path)
                }
            )
            
            self.images_info.append(image_info)
            category_space.append(category)
        
        self.dataset_info["category_space"] = list(set(category_space))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What emotion is expressed in the scene shown in this photo?"
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
    