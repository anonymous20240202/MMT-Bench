import os 
from pathlib import Path

import mmcv
from base_dataset import BaseDataset


class medical_modality_recognition(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/taskonomy_data/medical_understanding/modality_recognition",
        "anno_path": "/path/to/taskonomy_data/medical_understanding/modality_recognition/metadata_info.json",
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
            image_info["source"] = "medical_modality_recognition"

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import json
        import mmcv

        num_choices = 4
        question = image_info["question"]

        all_chocies = [image_info["option_A"], image_info["option_B"], image_info["option_C"], image_info["option_D"],]
        all_chocies.remove(image_info["gt_answer"])

        qa_json = {
            "num_wrong_choices": num_choices - 1,
            "gt": image_info["gt_answer"],
            "question": question,
            "wrong_choices_list": all_chocies
        }
        qa_json = BaseDataset.post_process(qa_json, question=question)

        return qa_json