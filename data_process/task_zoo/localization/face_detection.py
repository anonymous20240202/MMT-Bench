from pathlib import Path
from collections import defaultdict

import mmcv

from base_dataset import BaseDataset



class FDDB(BaseDataset):
    @staticmethod
    def process_raw_metadata_info(image_info):

        if len(image_info["bounding_box_coordinates"]) > 4:
            return False
        
        return image_info
    
    @staticmethod
    def generate_qa(image_info, dataset_info, _):
        from prompt.utils import generate_incorrect_bounding_boxes
        num_choices = 4
        width, height = image_info["width"], image_info["height"]
        question = f"Please detect all the faces in this image. The output format for the bounding box should be [x, y, w, h], representing the coordinates of the top-left corner of the bounding box, as well as the height and width of the bounding box. The width of the input image is {width} and the height is {height}."
        gt = image_info["bounding_box_coordinates"]

        new_gt = []
        for bbox in gt:
            new_gt.append([max(0, bbox[0]), max(0, bbox[1]), bbox[2], bbox[3]])

        while True:
            try:
                wrong_choices_list = generate_incorrect_bounding_boxes(new_gt, width, height, num_choices-1)
                qa_json = {
                    "question": question,
                    "num_wrong_choices": num_choices - 1,
                    "gt": new_gt, 
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                pass

        return qa_json
    

class WIDERFACE(BaseDataset):
    @staticmethod
    def process_raw_metadata_info(image_info):

        if len(image_info["bounding_box_coordinates"]) > 4:
            return False
        return image_info
    
    @staticmethod
    def generate_qa(image_info, dataset_info, _):
        from prompt.utils import generate_incorrect_bounding_boxes
        num_choices = 4
        width, height = image_info["width"], image_info["height"]
        question = f"Please detect all the faces in this image. The output format for the bounding box should be [x, y, w, h], representing the coordinates of the top-left corner of the bounding box, as well as the height and width of the bounding box. The width of the input image is {width} and the height is {height}."
        gt = image_info["bounding_box_coordinates"]

        new_gt = []
        for bbox in gt:
            new_gt.append([max(0, bbox[0]), max(0, bbox[1]), bbox[2], bbox[3]])

        while True:
            try:
                wrong_choices_list = generate_incorrect_bounding_boxes(new_gt, width, height, num_choices-1)
                qa_json = {
                    "question": question,
                    "num_wrong_choices": num_choices - 1,
                    "gt": new_gt, 
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                pass

        return qa_json
