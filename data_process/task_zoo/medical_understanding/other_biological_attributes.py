import os 
from pathlib import Path

import mmcv
from base_dataset import BaseDataset


class other_biological_attributes(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/taskonomy_data/medical_understanding/other_biological_attributes",
        "anno_path": "/path/to/taskonomy_data/medical_understanding/other_biological_attributes/metadata_info.json",
        "sampling_num": 200,
        "visual_input_component": ["medical_image", ],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        metadata_info = mmcv.load(self.anno_path)

        for image_info in metadata_info["images"]:
            image_info["original_image_path"] = os.path.join(self.image_path, image_info["image_path"])
            image_info["source"] = "other_biological_attributes"

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):

        num_choices = len([k for k in image_info.keys() if k.startswith('option')])
        question = image_info["question"]

        all_chocies = [v for k, v in image_info.items() if k.startswith('option')]
        all_chocies.remove(image_info["gt_answer"])

        qa_json = {
            "num_wrong_choices": num_choices - 1,
            "gt": image_info["gt_answer"],
            "question": question,
            "wrong_choices_list": all_chocies
        }
        qa_json = BaseDataset.post_process(qa_json, question=question)

        return qa_json
