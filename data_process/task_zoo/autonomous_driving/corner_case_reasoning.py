from pathlib import Path
from collections import OrderedDict

import mmcv

from base_dataset import BaseDataset


class CODA(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/taskonomy_data/autonomous_driving/corner_case_reasoning/metadata_info.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "NuScenes is an autonomous vehicle dataset, containing multi-view camera images captured by an autonomous driving car and annotation of traffic participants in forms of 2D boundary boxes and category. The scenes cover a decent variety of road scenes."
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)

        category_space = []

        for image_info in anno_info["images"]:
            image_info["original_image_path"] = image_info["image_path"].replace("lustre", "petrelfs")
            
            category = image_info["image_label"]

            image_info["category"] = category
            image_info["source"] = self.dataset_name

            category_space.append(category)
            self.images_info.append(image_info)

        self.dataset_info["category_space"] = list(set(category_space))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = image_info["question"]
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "There are 4 vehicle around the autonomous vehicle.",
                "question": question,
                "wrong_choices_list": ["There are 5 vehicle around the autonomous vehicle.", "There are 3 vehicle around the autonomous vehicle.", "There are no vehicle around the autonomous vehicle."]
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



class waymo_multiview(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/taskonomy_data/autonomous_driving/multi_view_images_reasoning/waymo/metadata_info.json",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "Waymo is an autonomous vehicle dataset, containing multi-view camera images captured by an autonomous driving car and annotation of traffic participants in forms of 2D boundary boxes and category. The scenes cover a decent variety of road scenes."
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)

        category_space = []

        for image_info in anno_info["images"]:
            image_info["original_image_path"] = image_info["image_path"].replace("lustre", "petrelfs")
            
            category = image_info["image_label"]

            image_info["category"] = category
            image_info["source"] = self.dataset_name

            category_space.append(category)
            self.images_info.append(image_info)

        self.dataset_info["category_space"] = list(set(category_space))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = image_info["question"]
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "There are 4 vehicle around the autonomous vehicle.",
                "question": question,
                "wrong_choices_list": ["There are 5 vehicle around the autonomous vehicle.", "There are 3 vehicle around the autonomous vehicle.", "There are no vehicle around the autonomous vehicle."]
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
