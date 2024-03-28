from pathlib import Path
from collections import defaultdict

import mmcv

from base_dataset import BaseDataset


class sod4bird(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/sod/sod4bird/train/images",
        "anno_path": "/path/to/lvlm_evaluation/data_process/data/sod/sod4bird/train/annotations/split_val_coco.json",
        "sampling_num": 100,
        "visual_input_component": ["natural_image", "high_resolution"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        anno_info = mmcv.load(self.anno_path)

        image2annotations = defaultdict(list)

        for annotation_info in anno_info["annotations"]:
            image2annotations[annotation_info["image_id"]].append(annotation_info)

        for image_anno_info in anno_info["images"]:
            image_id = image_anno_info["id"]
            if len(image2annotations[image_id]) >= 4:
                continue
            if len(image2annotations[image_id]) == 0:
                continue

            bounding_box_coordinates = []
            for anno in image2annotations[image_id]:
                bounding_box_coordinates.append([int(_) for _ in anno["bbox"]])
            
            image_info = self.image_dict(str(Path(self.image_path) / image_anno_info["file_name"]))
            image_info.update(
                {
                    "original_image_path": str(Path(self.image_path) / image_anno_info["file_name"]),
                    "bounding_box_coordinates": bounding_box_coordinates,
                    "type": "xywh"
                }
            )
            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, _):
        import json
        import mmcv
        from prompt.utils import generate_incorrect_bounding_boxes


        num_choices = 4
        width, height = image_info["width"], image_info["height"]
        question = f"Please detect all the birds in this image. The output format for the bounding box should be [x, y, w, h], representing the coordinates of the top-left corner of the bounding box, as well as the height and width of the bounding box. The width of the input image is {width} and the height is {height}."
        gt = image_info["bounding_box_coordinates"]

        while True:
            try:
                wrong_choices_list = generate_incorrect_bounding_boxes(gt, width, height, num_choices-1)
                qa_json = {
                    "question": question,
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt, 
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json)
                break
            except:
                pass

        return qa_json


class drone2021(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/sod/drone2021/images",
        "anno_path": "/path/to/lvlm_evaluation/data_process/data/sod/drone2021/annotations/split_val_coco.json",
        "sampling_num": 50,
        "visual_input_component": ["natural_image", "high_resolution"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        anno_info = mmcv.load(self.anno_path)

        image2annotations = defaultdict(list)

        for annotation_info in anno_info["annotations"]:
            image2annotations[annotation_info["image_id"]].append(annotation_info)

        for image_anno_info in anno_info["images"]:
            image_id = image_anno_info["id"]
            if len(image2annotations[image_id]) >= 4:
                continue
            if len(image2annotations[image_id]) == 0:
                continue
            bounding_box_coordinates = []
            for anno in image2annotations[image_id]:
                bounding_box_coordinates.append([int(_) for _ in anno["bbox"]])
            try:
                image_info = self.image_dict(str(Path(self.image_path) / image_anno_info["file_name"]))
            except:
                continue
                
            import os
            if not os.path.exists(str(Path(self.image_path) / image_anno_info["file_name"])):
                continue
            image_info.update(
                {
                    "original_image_path": str(Path(self.image_path) / image_anno_info["file_name"]),
                    "bounding_box_coordinates": bounding_box_coordinates,
                    "type": "xywh"
                }
            )
            self.images_info.append(image_info)
        
    @staticmethod
    def generate_qa(image_info, dataset_info, _):
        from prompt.utils import generate_incorrect_bounding_boxes

        num_choices = 4
        width, height = image_info["width"], image_info["height"]
        question = f"Please detect all the birds in this image. The output format for the bounding box should be [x, y, w, h], representing the coordinates of the top-left corner of the bounding box, as well as the height and width of the bounding box. The width of the input image is {width} and the height is {height}."
        gt = image_info["bounding_box_coordinates"]

        while True:
            try:
                wrong_choices_list = generate_incorrect_bounding_boxes(gt, width, height, num_choices-1)
                qa_json = {
                    "question": question,
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt, 
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json)
                break
            except:
                pass

        return qa_json


class tinyperson(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/sod/tiny_set/test",
        "anno_path": "/path/to/lvlm_evaluation/data_process/data/sod/tiny_set/annotations/tiny_set_test.json",
        "sampling_num": 50,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        anno_info = mmcv.load(self.anno_path)

        image2annotations = defaultdict(list)

        for annotation_info in anno_info["annotations"]:
            image2annotations[annotation_info["image_id"]].append(annotation_info)

        for image_anno_info in anno_info["images"]:
            image_id = image_anno_info["id"]
            if len(image2annotations[image_id]) >= 4:
                continue
            if len(image2annotations[image_id]) == 0:
                continue
            bounding_box_coordinates = []
            for anno in image2annotations[image_id]:
                bounding_box_coordinates.append([int(_) for _ in anno["bbox"]])
            
            image_info = self.image_dict(str(Path(self.image_path) / image_anno_info["file_name"]))
            image_info.update(
                {
                    "original_image_path": str(Path(self.image_path) / image_anno_info["file_name"]),
                    "bounding_box_coordinates": bounding_box_coordinates,
                    "type": "xywh"
                }
            )
            self.images_info.append(image_info)   
    
    @staticmethod
    def generate_qa(image_info, dataset_info, _):
        from prompt.utils import generate_incorrect_bounding_boxes

        num_choices = 4
        width, height = image_info["width"], image_info["height"]
        question = f"Please detect all the person in this image. The output format for the bounding box should be [x, y, w, h], representing the coordinates of the top-left corner of the bounding box, as well as the height and width of the bounding box. The width of the input image is {width} and the height is {height}."
        gt = image_info["bounding_box_coordinates"]

        while True:
            try:
                wrong_choices_list = generate_incorrect_bounding_boxes(gt, width, height, num_choices-1)
                qa_json = {
                    "question": question,
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt, 
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json)
                break
            except:
                pass

        return qa_json
