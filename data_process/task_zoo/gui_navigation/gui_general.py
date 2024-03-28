import os 
from pathlib import Path

import mmcv
from base_dataset import BaseDataset


class gui_general(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/taskonomy_data/gui_navigation/general/metadata_info.json",
        "sampling_num": 200,
        "visual_input_component": ["screenshot_image", ],
        "dataset_description": "GENERAL contains miscellaneous tasks (e.g., \u201cplay the new Taylor Swift video on YouTube\u201d), mostly centered around question and answering (Q & A) (e.g., \u201cHow much does a 2 bedroom apartment rent cost in San Francisco?\u201d) and interacting with third-party apps and websites."
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        metadata_info = mmcv.load(self.anno_path)

        for image_info in metadata_info["images"]:
            image_info["original_image_path"] = image_info["image_path"]
            image_info["original_image_path"] = image_info["original_image_path"].replace("/path/to/gui_data/aitw_pt", "/path/to/gui_dataset/aitw/pt")
            image_info["source"] = "gui_general"

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
            # "example_dict": {
            #     "num_wrong_choices": num_choices - 1,
            #     "gt": "Click at position [0.44, 0.92]",
            #     "question": question,
            #     "wrong_choices_list": ["Click at position [0.21, 0.83]", "Click at position [0.91, 0.24]", "Click at position [0.18, 0.25]"]
            # },
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
