from pathlib import Path
import csv

import mmcv
from PIL import Image

from base_dataset import BaseDataset


class python_auto_generated_color_name(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/color_recognition/colors.csv",
        "save_image_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/color_recognition/images",
        "sampling_num": 100,
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        csv_reader = csv.reader(open(self.anno_path))
        self.category_space = []
        self.color_rgb = {}

        for row in csv_reader:
            _, color_name, _, r, g, b = row

            image = Image.new("RGB", (200, 200), (int(r), int(g), int(b)))
            self.exist_or_mkdir(Path(self.save_image_path) )
            new_image_name = str(Path(self.save_image_path) / self.new_image_name(e='png'))
            image.save(new_image_name)
            rgb_category = (int(r), int(g), int(b))

            image_info = self.image_dict(new_image_name)
            image_info.update(
                {
                    "original_image_path": new_image_name,
                    "rgb_category": rgb_category,
                    "color_name":color_name
                }
            )
            self.category_space.append(color_name)
            self.color_rgb[color_name] = rgb_category
            self.images_info.append(image_info)

        self.dataset_info["category_space"] = list(set(self.category_space))
        self.dataset_info["color_rgb"] = self.color_rgb
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the color name in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Electric Lavender",
                "question": question,
                "wrong_choices_list": ["Rose Ebony", "Tufts Blue", "Blue Sapphire"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["color_name"],
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)

        while True:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                pass

        return qa_json


class python_auto_generated_color_rgb(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/color_recognition/colors.csv",
        "save_image_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/color_recognition/images",
        "sampling_num": 100,
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        
    
    def parse_images_info(self):
        self.images_info = list()
        csv_reader = csv.reader(open(self.anno_path))
        self.category_space = []
        self.color_rgb = {}

        for row in csv_reader:
            _, color_name, _, r, g, b = row

            image = Image.new("RGB", (200, 200), (int(r), int(g), int(b)))
            self.exist_or_mkdir(Path(self.save_image_path) )
            new_image_name = str(Path(self.save_image_path) / self.new_image_name(e='png'))
            image.save(new_image_name)
            rgb_category = (int(r), int(g), int(b))

            image_info = self.image_dict(new_image_name)
            image_info.update(
                {
                    "original_image_path": new_image_name,
                    "rgb_category": rgb_category,
                    "color_name":color_name
                }
            )
            self.category_space.append(color_name)
            self.color_rgb[color_name] = rgb_category
            self.images_info.append(image_info)

        self.dataset_info["category_space"] = list(set(self.category_space))
        self.dataset_info["color_rgb"] = self.color_rgb

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the color name in the picture?"
        _question = "What is the color RGB value (0~255) in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Electric Lavender",
                "question": question,
                "wrong_choices_list": ["Rose Ebony", "Tufts Blue", "Blue Sapphire"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["color_name"],
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)

        while True:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)


                qa_json["gt"] = dataset_info["color_rgb"][qa_json["gt"]]
                qa_json["wrong_choices_list"] = [dataset_info["color_rgb"][name] for name in qa_json["wrong_choices_list"]]
                qa_json = BaseDataset.post_process(qa_json, question=_question)
                break
            except:
                pass

        return qa_json
