import random
from pathlib import Path

from base_dataset import BaseDataset


class twod_geometric_shapes_dataset(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/shape/2D geometric shapes dataset/output",
        "sampling_num": 100,
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for image_path in Path(self.image_path).iterdir():
            shape_category = image_path.name.split("_")[0]
            self.category_space.append(shape_category)

            image_info = self.image_dict(str(image_path))
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "category": shape_category
                }
            )
            
            self.images_info.append(image_info)

        self.dataset_info["category_space"] = list(set(self.category_space))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the shape of this pattern in this image?"

        while True:
            try:
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": image_info["category"],
                    "question": question,
                    "wrong_choices_list": random.sample(dataset_info["category_space"], num_choices - 1)
                }
                qa_json = BaseDataset.post_process(qa_json)
                break
            except:
                pass

        return qa_json


class gpt_auto_generated_shape(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/shape/gpt_auto_generate_shape/images",
        "sampling_num": 100,
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.category_space = ["Circle", "Diamond", "Ellipse", "Heart Shape", "Hexagon/Pentagon"]
        self.dataset_info["category_space"] = self.category_space
    
    def parse_images_info(self):
        self.images_info = list()

        for image_path in Path(self.image_path).iterdir():
            shape_category = image_path.name.split("_")[0]
            if shape_category == "Pentagon":
                shape_category = "Hexagon/Pentagon"

            image_info = self.image_dict(str(image_path))
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "category": shape_category
                }
            )
            self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the shape of this major object in this image?"

        while True:
            try:
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": image_info["category"],
                    "question": question,
                    "wrong_choices_list": random.sample(dataset_info["category_space"], num_choices - 1)
                }
                qa_json = BaseDataset.post_process(qa_json)
                break
            except:
                pass

        return qa_json
    