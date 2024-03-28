from pathlib import Path
import csv
import json

import mmcv
import numpy as np
from PIL import Image
from iso3166 import countries

from base_dataset import BaseDataset


class funsd(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/ocr/data/VIE/FUNSD/images",
        "anno_path": "/path/to/lvlm_evaluation/data/ocr/data/VIE/FUNSD/en.val.kv.json",
        "sampling_num": 200,
        "visual_input_component": ["text-rich_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        anno_info_lsit = mmcv.load(self.anno_path)
        # cate_anno_list = mmcv.list_from_file(self.category_map)
        for image_name, anno_info in anno_info_lsit.items():
            pass
            image_path = Path(self.image_path) / image_name

            image_info = self.image_dict(image_path)
            image_info.update(
                {
                    "key_information": anno_info,
                    "original_image_path": str(image_path)
                }
            )
            self.images_info.append(image_info)


class sroie(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/ICDAR-2019-SROIE/data/img",
        "anno_path": "/path/to/lvlm_evaluation/data_process/data/ICDAR-2019-SROIE/data/key",
        "sampling_num": 200,
        "visual_input_component": ["text-rich_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        # anno_info_lsit = mmcv.load(self.anno_path)

        for image_path in Path(self.image_path).iterdir():
            anno_path = Path(self.anno_path) / f'{image_path.stem}.json'
            anno_info = mmcv.load(anno_path)

            image_info = self.image_dict(image_path)
            image_info.update(
                {
                    "key_information": anno_info,
                    "original_image_path": str(image_path),
                }
            )
            self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import mmcv
        import numpy as np
        import json
        import random
        num_choices = 4
        # question = "What is the interaction relationship between these two people (marked as RED bounding boxes)?"

        keys = image_info["key_information"].keys()
        key = random.choice(list(keys))

        question = f"Please read the text in this image and return the information about following key:\nKEY: {key}"

        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/general_prompt.txt'))

        input_json = {
            "question": question,
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "22/01/2017",
                "question": "Please read the text in this image and return the information about following key:\nKEY: date",
                "wrong_choices_list": ["27/01/2017", "22/01/2018", "22/06/2017"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["key_information"][key],
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