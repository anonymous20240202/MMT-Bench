from pathlib import Path
from collections import OrderedDict

from base_dataset import BaseDataset

import mmcv
import os

from tqdm import tqdm


class LVLM_eHub_conv_data(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/LVLM-eHub_conv_data/image_file_collect.pkl",
        "anoo_path": "/path/to/LVLM-eHub_conv_data/valid_conv_data.json",
        "save_image_path": "/path/to/lvlm_evaluation/data_process/taskonomy_evaluation_data/image_evaluation_judgement/lvlm_response_judgement/images",
        "sampling_num": 200,
        "url": "internet",
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.dataset_info["category_space"] = ["left", "right", "tie", "bad"]
    
    def parse_images_info(self):
        self.images_info = []
        image_anno_info = mmcv.load(self.image_path)
        conv_anno_info_list = mmcv.load(self.anoo_path)
        self.exist_or_mkdir(self.save_image_path)

        for conv_info in tqdm(conv_anno_info_list):
            image_name = conv_info["image"]
            save_image_path = os.path.join(self.save_image_path, image_name)
            
            # image_pil = image_anno_info['image_files'][image_anno_info['image_file_map'][image_name]]
            # image_pil.save(save_image_path)
            question = conv_info["question"]
            model_outputs = conv_info["model_outputs"]
            user_vote = conv_info["user_vote"]

            image_info = self.image_dict("")

            if not question:
                continue

            string_len = len(question) + sum([len(_) for _ in model_outputs])

            if string_len > 2000:
                continue
            image_info.update(
                {
                    "original_image_path": save_image_path,
                    "question": question,
                    "model_outputs": model_outputs,
                    "user_vote": user_vote
                }
            )
            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        width = image_info["width"]
        height = image_info["height"]
        question = f"I will provide you with an image along with a question related to that image. Additionally, there will be two possible answers to choose from. Your task is to evaluate and determine which answer is better, or if it's a tie, or if both answers are inadequate.\nQuestion: {image_info['question']}\nAnswer 1: {image_info['model_outputs'][0]}\nAnswer 1: {image_info['model_outputs'][1]}"

        if image_info['user_vote'] == "bad":
            wrong_choices_list = ['Answer 1', 'Answer 2', 'Tie']
            gt = "Both answers are bad"
        if image_info['user_vote'] == "tie":
            wrong_choices_list = ['Answer 1', 'Answer 2', 'Both answers are bad']
            gt = "Tie"
        if image_info['user_vote'] == "left":
            wrong_choices_list = ['Answer 2', 'Tie', 'Both answers are bad']
            gt = 'Answer 1'
        if image_info['user_vote'] == "right":
            wrong_choices_list = ['Answer 1', 'Tie', 'Both answers are bad']
            gt = 'Answer 2'
        

        qa_json = {
            "num_wrong_choices": num_choices - 1,
            "gt": gt,
            "question": question,
            "wrong_choices_list": wrong_choices_list
        }
        qa_json = BaseDataset.post_process(qa_json, question=question)

        return qa_json
        