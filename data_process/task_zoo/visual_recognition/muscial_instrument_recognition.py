from pathlib import Path
from collections import OrderedDict

from base_dataset import BaseDataset


class musical_instruments_image_classification(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/Music_Instruments/dataset_a/test",
        "sampling_num": 100,
        "url": "https://www.kaggle.com/datasets/gpiosenka/musical-instruments-image-classification/data",
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
        question = "What category of musical instrument is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "tuba",
                "question": question,
                "wrong_choices_list": ["harmonica", "guitar", "violen"]
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


class music_instruments_classification(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/Music_Instruments/dataset_b/instruments_data",
        "sampling_num": 100,
        "url": "https://www.kaggle.com/datasets/aayushme/music-instruments-classification/data",
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
        question = "What category of musical instrument is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "piano",
                "question": question,
                "wrong_choices_list": ["violen", "bass_guitar", "tabla"]
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
