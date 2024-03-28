import os 
from pathlib import Path

import mmcv
from base_dataset import BaseDataset


class gui_install(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/taskonomy_data/gui_navigation/install/metadata_info.json",
        "sampling_num": 200,
        "visual_input_component": ["screenshot_image", ],
        "dataset_description": "INSTALL contains high-level tasks related to installing and uninstalling apps, app login, and app login support (e.g., \"forgot password\") for 88 different apps available on the Google Play store."
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        metadata_info = mmcv.load(self.anno_path)

        for image_info in metadata_info["images"]:
            image_info["original_image_path"] = image_info["image_path"]
            image_info["original_image_path"] = image_info["original_image_path"].replace("/path/to/gui_data/aitw_pt", "/path/to/gui_dataset/aitw/pt")
            image_info["source"] = "gui_install"

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import mmcv
        import numpy as np
        import json
        import random
        num_choices = 4

        question = image_info["question"]
        gt = image_info["gt_answer"]

        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/general_prompt.txt'))

        input_json = {
            "question": question,
            "example_dict1": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Click at position [0.44, 0.92]",
                "question": question,
                "wrong_choices_list": ["Click at position [0.21, 0.83]", "Click at position [0.91, 0.24]", "Click at position [0.18, 0.25]"]
            },
            "example_dict2": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Lift from position [0.08, 0.06] to position [0.08, 0.06]",
                "question": question,
                "wrong_choices_list": [
                    "Lift from position [0.23, 0.23] to position [0.19, 0.98]", 
                    "Lift from position [0.90, 0.10] to position [0.20, 0.19]", 
                    "Lift from position [0.12, 0.88] to position [0.34, 0.29]"
                    ]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": gt,
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
