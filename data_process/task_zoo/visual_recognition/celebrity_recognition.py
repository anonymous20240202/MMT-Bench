import os
from pathlib import Path
from collections import OrderedDict

from base_dataset import BaseDataset

import mmcv


class lfw(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy/Vision Recognition-Celebrity",
        "anno_path": "/path/to/TaskOnomy/Vision Recognition-Celebrity/metadata_dataset.json",
        "sampling_num": 200,
        "url": "internet",
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)

        for image_info in anno_info["images"]:
            image_info["original_image_path"] = os.path.join(self.image_path, image_info["image_name"])
            image_info["source"] = self.dataset_name

            image_info["category"] = ' '.join(image_info["category"].split("_"))

            self.images_info.append(image_info)


        category_space = anno_info["dataset_list"][0]["category_space"]
        new_category_space = []
        for name in category_space:
            part = ' '.join(name.split("_"))
            new_category_space.append(part)
        self.dataset_info["category_space"] = new_category_space
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv
        import random

        num_choices = 4
        question = "What type of animal is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": random.sample(dataset_info["category_space"], 500),
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Julianne Moore",
                "question": question,
                "wrong_choices_list": ["Nate Blackwell", "Craig Burley", "Will Ferrell"]
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
