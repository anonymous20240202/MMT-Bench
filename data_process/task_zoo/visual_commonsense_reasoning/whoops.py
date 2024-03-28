from pathlib import Path
from collections import OrderedDict

from base_dataset import BaseDataset

import random
import mmcv


class whoops(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/lvlm_evaluation/Whoops/WHOOPS.json",
        "sampling_num": 200,
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        
        anno_info = mmcv.load(self.anno_path)

        for image_info in anno_info["images"]:
            self.images_info.append(image_info)

            image_info["source"] = self.dataset_name
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import mmcv
        import numpy as np
        import json
        import random
        import json
        import mmcv
        from openai import OpenAI
        from prompt.utils import encode_image_to_base64
        num_choices = 4

        # question = image_info["question"]
        # gt = image_info["gt_answer"]
        question = image_info["question"]
        answer = image_info["answer"]

        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/general_prompt.txt'))

        input_json = {
            "question": question,
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "stairs",
                "question": "The Statue of David has what placed on him?",
                "wrong_choices_list": ["leaves", "hot dog", "openai"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": answer,
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
