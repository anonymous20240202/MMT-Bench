import copy
from pathlib import Path

from base_dataset import BaseDataset


class FFplusplus(BaseDataset):
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        num_choices = 2
        question = "Is the face in the picture real or fake?"

        i = 0
        while i <= 10:
            try:
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": image_info["category"],
                    "question": question,
                    "wrong_choices_list": ["real" if image_info["category"] == "fake" else "fake"]
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json


class CelebDFv2(BaseDataset):
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        num_choices = 2
        question = "Is the face in the picture real or fake?"

        i = 0
        while i <= 10:
            try:
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": image_info["category"],
                    "question": question,
                    "wrong_choices_list": ["real" if image_info["category"] == "fake" else "fake"]
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json


class dalle_art_deepfake(BaseDataset):
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        num_choices = 2
        question = "Is the the picture human generated or AI generated?"

        i = 0
        while i <= 10:
            try:
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": "human generated" if image_info["category"] == "real" else "AI generated",
                    "question": question,
                    "wrong_choices_list": ["human generated" if image_info["category"] == "fake" else "AI generated"]
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json
            