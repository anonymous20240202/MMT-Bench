import os
import json
import sys
import csv
import numpy as np
from PIL import Image
from pathlib import Path

import mmcv
from tqdm import tqdm

from base_dataset import BaseDataset


class ferplus(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "data/face_emotion_recognition/ferplus/train.csv",
        "image_save_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/emotion/fer",
        "sampling_num": 100,
        "dataset_description": "xxx",
        "visual_input_component": ["natural_image", ]
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.dataset_info["category_space"] = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def parse_images_info(self):
        self.images_info = []
        # self.data_info = list()
        csv_reader = csv.reader(open(self.anno_path))
        next(csv_reader, None)
        for row in tqdm(csv_reader):
            category, imgbytes = row

            category = self.dataset_info["category_space"][int(category)]
            # 将像素值字符串拆分为数字列表，并将其转换为numpy数组
            pixel_values = np.array(imgbytes.split(), dtype=np.uint8)
            
            # 假设图像是48x48像素的灰度图像，可以根据需求调整尺寸和通道数
            image = pixel_values.reshape((48, 48))
            
            # 将numpy数组转换为Pillow图像对象
            image = Image.fromarray(image)

            # image_info["category"] = category

            new_image_name = os.path.join(self.image_save_path, self.new_image_name("png"))
            # image_info["image_file"] = save_image_file

            self.exist_or_mkdir(str(Path(self.image_save_path)))
            image.save(new_image_name)

            image_info = self.image_dict("")
            image_info.update(
                {
                    "original_image_path": new_image_name,
                    "category": category
                }
            )

            self.images_info.append(image_info)  
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What type of facial expression is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Angry",
                "question": question,
                "wrong_choices_list": ["Disgust", "Happy", "Fear"]
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


class facial_emotion_recognition_dataset(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/emotional_quotient_test/facial-emotion-recognition-dataset/data",
        "sampling_num": 100,
        "dataset_description": "xxx",
        "visual_input_component": ["natural_image", ]
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        # self.dataset_info["category_space"] = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def parse_images_info(self):
        self.images_info = []
        self.category_space = []

        for dir_path in Path(self.image_path).iterdir():
            for image_path in dir_path.iterdir():
                category = image_path.stem
                self.category_space.append(category)

                image_info = self.image_dict("")
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "category": category
                    }
                )

                self.images_info.append(image_info)

        self.dataset_info["category_space"] = list(set(self.category_space))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What type of facial expression is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Angry",
                "question": question,
                "wrong_choices_list": ["Disgust", "Happy", "Fear"]
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


# class ferg_db(BaseDataset):
#     METADATA_INFO = {
#         "image_path": "/path/to/lvlm_evaluation/data/face_emotion_recognition/FERG_DB_256",
#         "image_save_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/emotion/fer"
#     }
#     def __init__(self):
#         self.image_path = self.METADATA_INFO["image_path"]
#         self.image_save_path = self.METADATA_INFO["image_save_path"]

#         self.parse_dataset()
#         self.data_info = self.get_data_info()
    
#     def parse_dataset(self):
#         self.dataset_info = dict()
#         self.dataset_info["category_space"] = ["Angry", "Disgust", "Fear", "Joy", "Netural", "Sadness", "Surprise"]

#     def get_data_info(self):
#         self.data_info = list()
#         for actor_path in Path(self.image_path).iterdir():
#             for actor_emotion_path in actor_path.iterdir():
#                 for acrtor_emotion_image in actor_emotion_path.iterdir():
#                     ori_image_path = str(acrtor_emotion_image)
#                     actor, category = actor_emotion_path.name.split("_")
#                     category = category.capitalize()

#                     image_info = {}
#                     image_info["ori_image_path"] = ori_image_path
#                     image_info["actor"] = actor
#                     image_info["category"] = category
#                     image_info["source"] = "ferg_db"
#                     image_info["visual_input_component"] = ["synthetic_image"]

#                     self.data_info.append(image_info)
