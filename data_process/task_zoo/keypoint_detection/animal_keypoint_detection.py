import mmcv
import uuid
import sys
from pathlib import Path
sys.path.append("data_process")

from base_dataset import BaseDataset
import copy
import random
import numpy as np


def prepare_keypoint_sequence(keypoint_dict, visable=2):
    string_sequence = ""
    for key, value in keypoint_dict.items():
        
        if len(value) == 0 or value[-1] != visable:  
            string_sequence += f"{key}: INVISABLE |"
        else:
            string_sequence += f"{key}: {value[:2]} |"
    
    return string_sequence


def generate_negative_options(keypoint_dict, width, height, num_options, probability=0.2, visable=2):
    modified_keypoint_dicts = []

    max_offset = int(width / 15)

    for i in range(num_options):
        # 复制输入的关键点字典以进行修改
        modified_keypoint_dict = keypoint_dict.copy()

        # 随机选择一个关键点进行修改
        for keypoint, values in modified_keypoint_dict.items():
            if random.random() < probability:
                # 获取关键点的当前坐标和可见性
                if len(values) == 0:
                    continue
                current_x, current_y, current_visibility = values

                # 随机生成偏移量，可以根据需要修改范围
                offset_x = random.uniform(-max_offset, max_offset)
                offset_y = random.uniform(-max_offset, max_offset)

                # 计算新的坐标，同时确保不超出图片边缘
                new_x = max(0, min(current_x + offset_x, width))
                new_y = max(0, min(current_y + offset_y, height))

                # 随机生成新的可见性（1表示不可见，2表示可见）
                if int(current_visibility) == visable:
                    new_visibility = random.choices([0, visable], [1, 6], k=1)[0]
                else:
                    new_visibility = random.choices([visable, 0], [1, 100], k=1)[0]

                # 更新关键点的坐标和可见性
                modified_keypoint_dict[keypoint] = [round(new_x), round(new_y), new_visibility]


        # 将修改后的关键点字典添加到列表中
        modified_keypoint_dicts.append(modified_keypoint_dict)

    return modified_keypoint_dicts


class ap10k_keypoint(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to1/evaluation_data/finish/keypoint_detection/animal_keypoint_detection/metadata_info.json",
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
        for i in range(17):
            k_dict[keypoints["category_information"]['keypoints'][i]] = keypoints_dict[i*3:i*3+3]
        
        # generate_negative_options(k_dict, width, height, num_choices-1)

        question = f"please detect the keypoint of animal (marked as RED boxes) in this image. Each key point is represented in the form [x,y]. Note that the width of the input image is {width} and the height is {height}."

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
                qa_json["marked_image_path"] = new_image_path

                break
            except:
                pass
        
        return qa_json
        

class Animal_kingdom_keypoint(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to1/evaluation_data/finish/keypoint_detection/animal_keypoint_detection/metadata_info.json",
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
        for i in range(17):
            k_dict[keypoints["category_information"]['keypoints'][i]] = keypoints_dict[i*3:i*3+3]
        
        # generate_negative_options(k_dict, width, height, num_choices-1)

        question = f"please detect the keypoint of animal (marked as RED boxes) in this image. Each key point is represented in the form [x,y]. Note that the width of the input image is {width} and the height is {height}."

        BaseDataset.exist_or_mkdir(save_image_path)
        bbox_list = [np.array([[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]])]
        # width, height = image_info["width"], image_info["height"]
        # for box in image_info["bounding_box_coordinates"]:
        #     bbox_list.append([box[0] * width, box[1] * height, box[2] * width, box[3] * height])
        new_image_path = str(Path(save_image_path) / BaseDataset.new_image_name())
        mmcv.imshow_bboxes(image_info["original_image_path"], bbox_list, show=False, out_file=new_image_path, thickness=2, colors="red")

        while True:
            try:
                wrong_choices_list = generate_negative_options(k_dict, width, height, num_choices - 1, visable=1)
                gt_string = prepare_keypoint_sequence(k_dict, visable=1)
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
                qa_json["marked_image_path"] = new_image_path

                break
            except:
                pass
        
        return qa_json
