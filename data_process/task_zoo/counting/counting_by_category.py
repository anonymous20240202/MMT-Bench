from pathlib import Path

import mmcv
import numpy as np
from iso3166 import countries

from base_dataset import BaseDataset


def get_answer(gt, rangee=4, num=3):
    import random
    a = list(range(max(0, gt - rangee), gt + rangee))
    a.remove(gt)
    return random.sample(a, k=num)


class fsc147_category(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/counting/images_384_VarV2",
        "anno_path": "/path/to/lvlm_evaluation/data/counting/annotation_FSC147_384.json",
        "density_map": "/path/to/lvlm_evaluation/data/counting/gt_density_map_adaptive_384_VarV2",
        "category_map": "/path/to/lvlm_evaluation/data/counting/ImageClasses_FSC147.txt",
        "sampling_num": 200,
        "visual_input_component": ["natural_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        # anno_info_lsit = mmcv.load(self.anno_path)
        cate_anno_list = mmcv.list_from_file(self.category_map)
        for anno_line in cate_anno_list:
            image_name, category = anno_line.split('\t')

            counting_num = int(np.load((Path(self.density_map) / image_name).with_suffix('.npy')).sum())

            image_info = self.image_dict(str(Path(self.image_path) / image_name))
            image_info.update(
                {
                    "category": category,
                    "counting_num": counting_num,
                    "original_image_path": str(Path(self.image_path) / image_name)
                }
            )
            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info):
        category = image_info["category"]
        question = f"Please count how many {category} in this image."

        qa_json = {
            "question": question,
            "num_wrong_choices": 3,
            "wrong_choices_list": get_answer(image_info["counting_num"], rangee=4),
            "gt": image_info['counting_num']
        }

        while True:
            try:
                qa_json = BaseDataset.post_process(qa_json)
                break
            except:
                pass

        return qa_json


class countqa_vqa(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/coco/val2014",
        "anno_path": "/path/to/lvlm_evaluation/data/counting/coco_counting/CountQA_VQA_data.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        anno_info_lsit = mmcv.load(self.anno_path)

        for anno_info in anno_info_lsit:
            question = anno_info["Question"]
            answer = anno_info["Answer"]
            original_image_path = Path(self.image_path) / anno_info["Image_ID"]

            image_info = self.image_dict(str(original_image_path))
            image_info.update(
                {
                    "original_image_path": str(original_image_path),
                    "question": question,
                    "answer": answer
                }
            )

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info):
        question = image_info["question"]

        qa_json = {
            "question": question,
            "num_wrong_choices": 3,
            "wrong_choices_list": get_answer(image_info["answer"]),
            "gt": image_info["answer"]
        }

        while True:
            try:
                qa_json = BaseDataset.post_process(qa_json)
                break
            except:
                pass

        return qa_json


class countqa_cocoqa(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/coco/val2014",
        "anno_path": "/path/to/lvlm_evaluation/data/counting/coco_counting/CountQA_COCOQA_data.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        anno_info_lsit = mmcv.load(self.anno_path)

        for anno_info in anno_info_lsit:
            question = anno_info["Question"]
            answer = anno_info["Answer"]
            original_image_path = Path(self.image_path) / anno_info["Image_ID"]

            image_info = self.image_dict(str(original_image_path))
            image_info.update(
                {
                    "original_image_path": str(original_image_path),
                    "question": question,
                    "answer": answer
                }
            )

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info):
        question = image_info["question"]

        qa_json = {
            "question": question,
            "num_wrong_choices": 3,
            "wrong_choices_list": get_answer(image_info["answer"]),
            "gt": image_info["answer"]
        }

        while True:
            try:
                qa_json = BaseDataset.post_process(qa_json)
                break
            except:
                pass

        return qa_json
    

class tallyqa_simple(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/visual_genome",
        "anno_path": "/path/to/lvlm_evaluation/data/counting/tallyqa/test.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        anno_info_lsit = mmcv.load(self.anno_path)
        # cate_anno_list = mmcv.list_from_file(self.category_map)
        for anno_info in anno_info_lsit:
            if not anno_info["issimple"]:
                continue
            original_image_path = Path(self.image_path) / anno_info['image']
            answer = anno_info["answer"]
            question = anno_info["question"]

            image_info = self.image_dict(str(original_image_path))
            image_info.update(
                {
                    "question": question,
                    "answer": answer,
                    "original_image_path": str(original_image_path)
                }
            )
            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info):
        question = image_info["question"]

        qa_json = {
            "question": question,
            "num_wrong_choices": 3,
            "wrong_choices_list": get_answer(image_info["answer"]),
            "gt": image_info["answer"]
        }

        while True:
            try:
                qa_json = BaseDataset.post_process(qa_json)
                break
            except:
                pass

        return qa_json
