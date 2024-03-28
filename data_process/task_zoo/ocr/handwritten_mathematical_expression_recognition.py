from pathlib import Path

import struct
import numpy as np
import mmcv

from base_dataset import BaseDataset


class hme100k(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/ocr/data/HMER/HME100K/images",
        "anno_path": "/path/to/lvlm_evaluation/data/ocr/data/HMER/HME100K/GT.txt",
        "sampling_num": 100,
        "visual_input_component": ["text-rich_image", "natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_list = mmcv.list_from_file(self.anno_path)
        for image_anno in anno_list:
            image_name, text = image_anno.split('\t')

            original_image_path = Path(self.image_path) / image_name

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(original_image_path),
                    "text": text
                }
            )

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info):
        import json
        import mmcv

        num_choices = 4
        question = "Recognize handwritten mathematical expression in the image."
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/handwritten_mathematical_expression_recognition.txt'))

        input_json = {
            "question": question,
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "\\angle A F E = 3 0 ^ { \\circ } , A E = 1 2",
                "question": question,
                "wrong_choices_list": [
                        "\\angle AFE = 30^{\\circ}, AE = 21",
                        "\\angle AFE = 60^{\\circ}, AE = 12",
                        "\\angle AFE = 30^{\\circ}, AF = 12"
                    ]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["text"],
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)

        while True:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json)
                break
            except:
                pass

        return qa_json


class crohme2014(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/ocr/data/HMER/CROHME2014/images",
        "anno_path": "/path/to/lvlm_evaluation/data/ocr/data/HMER/CROHME2014/GT.txt",
        "sampling_num": 100,
        "visual_input_component": ["synthetic_image", "text-rich_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_list = mmcv.list_from_file(self.anno_path)
        for image_anno in anno_list:
            image_name, text = image_anno.split('\t')

            original_image_path = Path(self.image_path) / image_name

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(original_image_path),
                    "text": text
                }
            )

            self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info):
        import json
        import mmcv

        num_choices = 4
        question = "Recognize handwritten mathematical expression in the image."
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/handwritten_mathematical_expression_recognition.txt'))

        input_json = {
            "question": question,
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "x ^ { 2 } + 5 / 6 x + 1 / 6",
                "question": question,
                "wrong_choices_list": [
                        "x^{2} + \\frac{5}{6}x - \\frac{1}{6}",
                        "x^{2} - \\frac{5}{6}x + \\frac{1}{6}",
                        "x^{2} + \\frac{6}{5}x + \\frac{1}{6}"
                    ]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["text"],
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)

        while True:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json)
                break
            except:
                pass

        return qa_json
    