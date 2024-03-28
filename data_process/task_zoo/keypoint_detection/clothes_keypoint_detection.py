import mmcv
import uuid
import sys
from pathlib import Path
sys.path.append("data_process")

from base_dataset import BaseDataset
import copy

from .furniture_keypoint_detection import *


class Deepfashion_keypoint(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to1/evaluation_data/finish/keypoint_detection/clothes_keypoint_detection/metadata_info.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image", ],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        metadata_info = mmcv.load(self.anno_path)

        for image_info in metadata_info["images"]:
            if "/Deepfashion/" not in image_info["file_name"]:
                continue
            image_info["source"] = self.dataset_name

            for keypoint_info in image_info["key_ann"]:

                _image_info = copy.deepcopy(image_info)
                _image_info["keypoints"] = keypoint_info

                _image_info["original_image_path"] = _image_info["file_name"]

                self.images_info.append(_image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        num_choices = 4
        width, height = image_info["width"], image_info["height"]
        keypoints = image_info["keypoints"]

        keypoints_dict = keypoints["keypoints"]
        bbox = keypoints["bbox"]
        k_dict = {}
        for i in range(keypoints['num_keypoints']):
            k_dict[keypoints["category_information"]['keypoints'][i]] = keypoints_dict[i*3:i*3+3]
        
        # generate_negative_options(k_dict, width, height, num_choices-1)

        question = f"please detect the keypoints (marked as RED box) of clothes shown in this image. Each key point is represented in the form [x,y] if it is VISABLE. Note that the width of the input image is {width} and the height is {height}."

        BaseDataset.exist_or_mkdir(save_image_path)
        bbox_list = [np.array([np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])])]
        # width, height = image_info["width"], image_info["height"]
        # for box in image_info["bounding_box_coordinates"]:
        #     bbox_list.append([box[0] * width, box[1] * height, box[2] * width, box[3] * height])
        new_image_path = str(Path(save_image_path) / BaseDataset.new_image_name())
        mmcv.imshow_bboxes(image_info["original_image_path"], bbox_list, show=False, out_file=new_image_path, thickness=2, colors="red")

        while True:
            try:
                wrong_choices_list = generate_negative_options(k_dict, width, height, num_choices - 1)
                gt_string = prepare_keypoint_sequence(k_dict)
                new_wrong_list = []

                for wrong_choice in wrong_choices_list:
                    new_wrong_list.append(prepare_keypoint_sequence(wrong_choice))
                

                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt_string,
                    "question": question,
                    "wrong_choices_list": new_wrong_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                qa_json["original_image_path"] = new_image_path

                break
            except:
                pass
        
        return qa_json


class Deepfashion2_keypoint(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to1/evaluation_data/finish/keypoint_detection/clothes_keypoint_detection/metadata_info.json",
        "sampling_num": 100,
        "visual_input_component": ["natural_image", ],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        metadata_info = mmcv.load(self.anno_path)

        for image_info in metadata_info["images"]:
            if "/Deepfashion2/" not in image_info["file_name"]:
                continue
            image_info["source"] = self.dataset_name

            for keypoint_info in image_info["key_ann"]:

                _image_info = copy.deepcopy(image_info)
                _image_info["keypoints"] = keypoint_info

                import os
                if not os.path.exists(str(Path("/path/to1/evaluation_data/data/mp100/Deepfashion2") / Path(_image_info["file_name"]).name)):
                    continue
                _image_info["original_image_path"] = str(Path("/path/to1/evaluation_data/data/mp100/Deepfashion2") / Path(_image_info["file_name"]).name)

                self.images_info.append(_image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        num_choices = 4
        width, height = image_info["width"], image_info["height"]
        keypoints = image_info["keypoints"]

        keypoints_dict = keypoints["keypoints"]
        bbox = keypoints["bbox"]
        k_dict = {}
        for i in range(keypoints['num_keypoints']):
            k_dict[keypoints["category_information"]['keypoints'][i]] = keypoints_dict[i*3:i*3+3]
        
        # generate_negative_options(k_dict, width, height, num_choices-1)

        question = f"please detect the keypoints (marked as RED box) of clothes shown in this image. Each key point is represented in the form [x,y] if it is VISABLE. Note that the width of the input image is {width} and the height is {height}."

        BaseDataset.exist_or_mkdir(save_image_path)
        bbox_list = [np.array([np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])])]
        # width, height = image_info["width"], image_info["height"]
        # for box in image_info["bounding_box_coordinates"]:
        #     bbox_list.append([box[0] * width, box[1] * height, box[2] * width, box[3] * height])
        new_image_path = str(Path(save_image_path) / BaseDataset.new_image_name())
        mmcv.imshow_bboxes(image_info["original_image_path"], bbox_list, show=False, out_file=new_image_path, thickness=2, colors="red")

        while True:
            try:
                wrong_choices_list = generate_negative_options(k_dict, width, height, num_choices - 1)
                gt_string = prepare_keypoint_sequence(k_dict)
                new_wrong_list = []

                for wrong_choice in wrong_choices_list:
                    new_wrong_list.append(prepare_keypoint_sequence(wrong_choice))
                

                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt_string,
                    "question": question,
                    "wrong_choices_list": new_wrong_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                qa_json["original_image_path"] = new_image_path

                break
            except:
                pass
        
        return qa_json
