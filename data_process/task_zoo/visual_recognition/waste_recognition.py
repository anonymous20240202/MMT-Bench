from pathlib import Path

from base_dataset import BaseDataset


class Garbage_Classification_12(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/waste/Garbage_Classification_12/garbage_classification",
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
        question = "What category of the waste is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "biological",
                "question": question,
                "wrong_choices_list": ["plastic", "clothes", "paper"]
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


class Waste_Classification_data(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/waste/Waste_Classification_data/dataset/DATASET/TEST",
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
            category = {"O": "Organic", "R": "Recyclable"}[category_path.name]
            self.category_lsit.append(category)

            for image_path in category_path.iterdir():
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

        num_choices = 2
        question = "What category of the waste is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Organic",
                "question": question,
                "wrong_choices_list": ["Recyclable"]
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
            