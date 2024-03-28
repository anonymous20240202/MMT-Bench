from pathlib import Path
import csv
import os
import mmcv
from PIL import Image
import struct
import numpy as np

from base_dataset import BaseDataset

from prompt.utils import *


class rock_image_recognition1(BaseDataset):
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What category of the rock is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": [replace_underscore_with_space(i) for i in dataset_info["category_space"]],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "amphibolite",
                "question": question,
                "wrong_choices_list": ["basalt", "flint", "siltstone"]
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


class rock_image_recognition2(BaseDataset):
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What category of the rock is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": [replace_underscore_with_space(i) for i in dataset_info["category_space"]],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "basalt",
                "question": question,
                "wrong_choices_list": ["coal", "granite", "marble"]
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
