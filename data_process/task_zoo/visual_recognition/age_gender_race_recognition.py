import os
from pathlib import Path
from collections import OrderedDict

from base_dataset import BaseDataset

import mmcv


class FairFace(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy/Vision Recognition-Age, Gender or Race",
        "anno_path": "/path/to/TaskOnomy/Vision Recognition-Age, Gender or Race/metadata_dataset.json",
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

        category_age = []
        category_race = []
        category_gender = []

        for image_info in anno_info["images"]:

            image_info["original_image_path"] = os.path.join(self.image_path, image_info["image_name"])
            image_info["source"] = self.dataset_name

            category_age.append(image_info["age"])
            category_race.append(image_info["race"])
            category_gender.append(image_info["gender"])

            self.images_info.append(image_info)

        self.dataset_info["category_age"] = list(set(category_age))
        self.dataset_info["category_race"] = list(set(category_race))
        self.dataset_info["category_gender"] = list(set(category_gender))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv
        import random
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        key = random.choice(["age", "race", "gender"])
        if key == "age":
            num_choices = 4
            question = "How old is the people shown in this image?"
            gt = image_info["age"]
            category_space = dataset_info["category_age"]

            input_json = {
            "question": question,
                "category_space": category_space,
                "example_dict": {
                    "num_wrong_choices": num_choices - 1,
                    "gt": "0-2",
                    "question": question,
                    "wrong_choices_list": ["50-59", "60-69", "3-9"]
                },
                "query_dict": {
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt,
                    "question": question,
                }
            }
        elif key == "race":
            num_choices = 4
            question = "What race is the person in the picture?"
            gt = image_info["race"]
            category_space = dataset_info["category_race"]

            input_json = {
                "question": question,
                    "category_space": category_space,
                    "example_dict": {
                        "num_wrong_choices": num_choices - 1,
                        "gt": "Southeast Asian",
                        "question": question,
                        "wrong_choices_list": ["Black", "White", "Indian"]
                    },
                    "query_dict": {
                        "num_wrong_choices": num_choices - 1,
                        "gt": gt,
                        "question": question,
                    }
            }

        else:
            num_choices = 2
            question = "What gender is the person in the picture?"
            gt = image_info["gender"]
            category_space = dataset_info["category_gender"]
            input_json = {
                "question": question,
                    "category_space": category_space,
                    "example_dict": {
                        "num_wrong_choices": num_choices - 1,
                        "gt": "Male",
                        "question": question,
                        "wrong_choices_list": ["Female"]
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
