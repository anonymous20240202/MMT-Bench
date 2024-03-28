from pathlib import Path

from base_dataset import BaseDataset


class scitsr(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/ocr/data/TSR/SciTSR/images",
        "anno_path": "/path/to/lvlm_evaluation/data/ocr/data/TSR/SciTSR/GT",
        "sampling_num": 200,
        "visual_input_component": ["text-rich_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import mmcv
        self.images_info = list()

        for image_path in Path(self.image_path).iterdir():
            anno_path = (Path(self.anno_path) / image_path.name).with_suffix(".txt")
        
            text = mmcv.list_from_file(anno_path)[0]

            image_info = self.image_dict(image_path)
            image_info.update(
                {
                    "text": text,
                    "original_image_path": str(image_path)
                }
            )
            self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import mmcv
        import numpy as np
        import json
        import random

        question = f"Please read the table in this image and return a html-style reconstructed table in text, do not omit anything."
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/general_prompt.txt'))

        gt = image_info["text"]

        if len(gt) >= 1000:
            return None
        else:
            num_choices = 2
        input_json = {
            "question": question,
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "<html><body><table><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table></body></html>",
                "question": "Please read the table in this image and return a html-style reconstructed table in text, do not omit anything.",
                "wrong_choices_list": ["<html><body><table><tr><td></td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table></body></html>", "22/01/2018", "22/06/2017"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": gt,
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
        