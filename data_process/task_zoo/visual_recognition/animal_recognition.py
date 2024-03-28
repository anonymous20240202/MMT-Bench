from pathlib import Path
from collections import OrderedDict

from base_dataset import BaseDataset


class animals90(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/animals/Animal_Image_Dataset_90/animals/animals",
        "sampling_num": 50,
        "url": "internet",
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        self.category_lsit = []
        for category_path in Path(self.image_path).iterdir():
            category = category_path.name
            self.category_lsit.append(category)

            for image_path in Path(category_path).iterdir():

                image_info = self.image_dict(str(image_path))
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "catgeory": category
                    }
                )
                self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = list(set(self.category_lsit))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What type of animal is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "dog",
                "question": question,
                "wrong_choices_list": ["bear", "cat", "ladybugs"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["catgeory"],
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


class animals150(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/animals/animals150/dataset/dataset",
        "anno_path": "/path/to/lvlm_evaluation/data_process/data/animals/animals150/translation.json",
        "sampling_num": 100,
        "url": "internet",
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import mmcv
        self.images_info = list()
        category_map = mmcv.load(self.anno_path)

        self.category_lsit = []
        for category_path in Path(self.image_path).iterdir():
            category = category_map[category_path.name]
            self.category_lsit.append(category)

            for image_path in Path(category_path).iterdir():

                image_info = self.image_dict(str(image_path))
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "catgeory": category
                    }
                )
                self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = list(set(self.category_lsit))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What type of animal is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Gorilla",
                "question": question,
                "wrong_choices_list": ["Tiger", "Tapir", "Moose"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["catgeory"],
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
    

class animals10(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/animals/animals10/raw-img",
        "anno_path": "/path/to/lvlm_evaluation/data_process/data/animals/animals150/translation.json",
        "sampling_num": 50,
        "url": "internet",
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import mmcv
        self.images_info = list()
        category_map = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane", "cavallo": "horse", "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca", "ragno": "spider", "squirrel": "scoiattolo"}

        self.category_lsit = []
        for category_path in Path(self.image_path).iterdir():
            category = category_map[category_path.name]
            self.category_lsit.append(category)

            for image_path in Path(category_path).iterdir():

                image_info = self.image_dict(str(image_path))
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "catgeory": category
                    }
                )
                self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = list(set(self.category_lsit))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What type of animal is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "horse",
                "question": question,
                "wrong_choices_list": ["dog", "butterfly", "squirrel"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["catgeory"],
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
