from pathlib import Path

import numpy as np

from base_dataset import BaseDataset


class internet_poster(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/movie/internet/films",
        "sampling_num": 50,
        "url": "opendatalab",
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        for movie_category_path in Path(self.image_path).iterdir():
            movie_category = movie_category_path.name

            for movie_path in movie_category_path.iterdir():
                movie_name = movie_path.stem
                self.category_space.append(movie_name)

                image_info = self.image_dict(str(movie_path))
                image_info.update(
                    {
                        "original_image_path": str(movie_path),
                        "category": movie_name,
                        "movie_type": movie_category
                    }
                )

                self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = self.category_space
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the name of the television work in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Breaking Bad",
                "question": question,
                "wrong_choices_list": ["The Crown", "Jessica Jones", "12 Angry Men"]
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


class movie_posters_kaggle(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/movie/movie_posters/Multi_Label_dataset/Images",
        "anno_path": "/path/to/lvlm_evaluation/data/movie/movie_posters/Multi_Label_dataset/train.csv",
        "sampling_num": 150,
        "url": "opendatalab",
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import csv
        self.images_info = list()
        self.category_space = []

        anno_info_list = csv.reader(open(self.anno_path))
        next(anno_info_list)
        for row in anno_info_list:
            image_name, category_list = row[0], eval(row[1])

            if len(category_list) != 1:
                continue
            if category_list[0] == "N/A":
                continue

            original_image_path = str(Path(self.image_path) / f"{image_name}.jpg")

            image_info = self.image_dict(original_image_path)
            image_info.update(
                {
                    "original_image_path": original_image_path,
                    "category": category_list,
                }
            )

            self.images_info.append(image_info)

            self.category_space.extend(category_list)
        
        self.dataset_info["category_space"] = list(set(self.category_space))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the genre of the movie in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Horror",
                "question": question,
                "wrong_choices_list": ["Animation", "Thriller", "Sport"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["category"][0],
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
    