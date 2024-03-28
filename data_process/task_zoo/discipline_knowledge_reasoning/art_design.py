from pathlib import Path
from collections import OrderedDict

import mmcv

from base_dataset import BaseDataset

import pandas as pd
import pyarrow.parquet as pq

import os

from tqdm import tqdm

class mmmu_art_design(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/MMMU", 
        "sampling_num": 120,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }

    sub_subject_list = [
        "Art",
        "Art_Theory",
        "Design",
        "Music"
    ]
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        for sub_subject in self.sub_subject_list:
            anno_path = os.path.join(self.image_path, sub_subject, "validation-00000-of-00001.parquet")

            df = pd.read_parquet(anno_path)

            for _, row in tqdm(enumerate(df.itertuples())):
                if row.image_1 and not row.image_2:

                    if row.question_type != "multiple-choice":
                        continue

                    image = mmcv.imfrombytes(row.image_1["bytes"])

                    answer = row.answer
                    question = row.question
                    options = row.options

                    
                    original_image_path = os.path.join(self.image_path, sub_subject, self.new_image_name())

                    self.save_rgb_image(image, original_image_path)

                    image_info = self.image_dict("")
                    image_info.update(
                        {
                            "original_image_path": original_image_path,
                            "question": question,
                            "options": options,
                            "answer": answer
                        }
                    )

                    self.images_info.append(image_info)

    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        gt_index = ord(image_info["answer"]) - ord("A")

        options = eval(image_info["options"])

        gt = options[gt_index]
        wrong_choices_list = options
        wrong_choices_list.remove(gt)

        question = image_info["question"]
    
        qa_json = {
            "num_wrong_choices": len(wrong_choices_list),
            "gt": gt,
            "question": question,
            "wrong_choices_list": wrong_choices_list
        }

        i = 0
        while i <= 10:
            try:
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json
