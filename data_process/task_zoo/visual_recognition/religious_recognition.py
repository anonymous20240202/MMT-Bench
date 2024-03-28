from pathlib import Path
from collections import OrderedDict

from base_dataset import BaseDataset


class religious_symbols_image_classification(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/religious/dataset_a/Symbols",
        "sampling_num": 100,
        "url": "https://www.kaggle.com/datasets/kumarujjawal123456/famous-religious-symbols",
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        for image_category in Path(self.image_path).iterdir():
            category_name = image_category.name
            self.category_space.append(category_name)

            for image_path in image_category.iterdir():
                original_image_path = str(image_path)
                try:
                    image_info = self.image_dict(original_image_path)
                except:
                    continue
                image_info.update(
                    {
                        "original_image_path": original_image_path,
                        "category": category_name
                    }
                )

                self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = self.category_space

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the object in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Swastik",
                "question": question,
                "wrong_choices_list": ["All_seeing_eye", "Chi_Rho", "Dharma_chakra_Buddhism"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["category"],
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


class dataset_of_traditional_chinese_god_statue(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/religious/dataset_b/Dataset_of_traditional_Chinese_god_statue/traditional_Chinese_god_statue/Original",
        "sampling_num": 100,
        "url": "https://www.sciencedirect.com/science/article/pii/S2352340922010642",
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        for image_category in Path(self.image_path).iterdir():
            category_name = image_category.name
            self.category_space.append(category_name)

            for image_path in (image_category / "test").iterdir():
                original_image_path = str(image_path)
                try:
                    image_info = self.image_dict(original_image_path)
                except:
                    continue
                image_info.update(
                    {
                        "original_image_path": original_image_path,
                        "category": category_name
                    }
                )

                self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = self.category_space

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the object in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "forg",
                "question": question,
                "wrong_choices_list": ["bear", "dog", "cat"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["category"],
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
