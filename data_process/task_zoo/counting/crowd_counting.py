import os
from pathlib import Path
from collections import OrderedDict

from base_dataset import BaseDataset

import mmcv


class ShanghaiTech(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy/Crowd Couting",
        "anno_path": "/path/to/TaskOnomy/Crowd Couting/metadata_dataset.json",
        "sampling_num": 100,
        "url": "internet",
        "visual_input_component": ["natural_image"],
        "dataset_description": "The Shanghaitech dataset is a large-scale crowd counting dataset. It consists of 1198 annotated crowd images. The dataset is divided into two parts, Part-A containing 482 images and Part-B containing 716 images. Part-A is split into train and test subsets consisting of 300 and 182 images, respectively. Part-B is split into train and test subsets consisting of 400 and 316 images. Each person in a crowd image is annotated with one point close to the center of the head. In total, the dataset consists of 330,165 annotated people. Images from Part-A were collected from the Internet, while images from Part-B were collected on the busy streets of Shanghai.",
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)
        
        for image_info in anno_info["images"]:
            if image_info["source"] != "ShanghaiTech":
                continue
            if image_info["image_name"].startswith("Images"):
                continue
            image_path = os.path.join(self.image_path, "ShanghaiTech", image_info["image_name"])

            image_info["original_image_path"] = image_path

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import mmcv
        import json
        num_choices = 4

        question = "Please count the number of people in the picture."

        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/general_prompt.txt'))

        input_json = {
            "question": question,
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "156",
                "question": "Please read the text in this image and return the information about following key:\nKEY: date",
                "wrong_choices_list": ["140", "72", "66"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["count"],
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


class CARPK(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy/Crowd Couting/CARPK",
        "anno_path": "/path/to/TaskOnomy/Crowd Couting/CARPK/test.txt",
        "sampling_num": 100,
        "url": "internet",
        "visual_input_component": ["natural_image"],
        "dataset_description": "The Car Parking Lot Dataset (CARPK) contains nearly 90,000 cars from 4 different parking lots collected by means of drone (PHANTOM 3 PROFESSIONAL). The images are collected with the drone-view at approximate 40 meters height. The image set is annotated by bounding box per car. All labeled bounding boxes have been well recorded with the top-left points and the bottom-right points. It is supporting object counting, object localizing, and further investigations with the annotation format in bounding boxes.",
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info_list = mmcv.list_from_file(self.anno_path)

        for anno_info in anno_info_list:
            image_name, count = anno_info.split(" ")

            image_path = os.path.join(self.image_path, image_name)

            image_info = self.image_dict("")
            image_info.update(
                {
                    "original_image_path": image_path,
                    "count": int(count)
                }
            )

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import mmcv
        import json
        num_choices = 4

        question = "Please count the number of cars in the picture."

        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/general_prompt.txt'))

        input_json = {
            "question": question,
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "156",
                "question": "Please read the text in this image and return the information about following key:\nKEY: date",
                "wrong_choices_list": ["140", "72", "66"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["count"],
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
