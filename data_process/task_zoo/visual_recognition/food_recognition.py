from pathlib import Path

from base_dataset import BaseDataset


class Fruits_and_Vegetables(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/food/Fruits_and_Vegetables/test",
        "sampling_num": 100,
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
        
        self.dataset_info['category_space'] = list(set(self.category_lsit))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What category of food is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "pineapple",
                "question": question,
                "wrong_choices_list": ["spinach", "sweetpotato", "mango"]
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


class food_101(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/food/food_101/food-101/images",
        "anno_path": "/path/to/lvlm_evaluation/data_process/data/food/food_101/food-101/meta/test.txt",
        "sampling_num": 100,
        "url": "internet",
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import mmcv
        import os
        self.images_info = list()

        self.category_lsit = []
        anno_list = mmcv.list_from_file(self.anno_path)

        for image_file in anno_list:
            category = ' '.join(image_file.split('/')[0].split('_'))
            self.category_lsit.append(category)

            image_path = os.path.join(self.image_path, f'{image_file}.jpg')

            image_info = self.image_dict(str(image_path))

            image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "catgeory": category
                    }
                )
            self.images_info.append(image_info)
        
        self.dataset_info['category_space'] = list(set(self.category_lsit))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What category of food is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "ramen",
                "question": question,
                "wrong_choices_list": ["steak", "fried rice", "pho"]
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
            