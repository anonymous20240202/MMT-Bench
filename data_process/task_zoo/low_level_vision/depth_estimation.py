from pathlib import Path

import mmcv

from base_dataset import BaseDataset


class taskonomy(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/depth_data/taskonomy",
        "sampling_num": 50,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info_list = mmcv.load(Path(self.image_path) / "annotations.json")

        for anno_info in anno_info_list:
            rgb_path = Path(self.image_path) / "rgb" / anno_info["rgb"]
            depth_path = Path(self.image_path) / "depth" / anno_info["depth"]

            image_info = self.image_dict("")
            image_info.update(
                {
                    "original_image_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "cam_in": anno_info["cam_in"],
                    "depth_metric": anno_info["depth_metric"],
                    "max": anno_info["max"],
                }
            )

            self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the object in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "forg",
                "question": question,
                "wrong_choices_list": ["bear", "dog", "cat"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["category"],
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)

        while True:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json)
                break
            except:
                pass

        return qa_json

class nyu(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/depth_data/nyu",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info_list = mmcv.load(Path(self.image_path) / "annotations.json")

        for anno_info in anno_info_list:
            rgb_path = Path(self.image_path) / "rgb" / anno_info["rgb"]
            depth_path = Path(self.image_path) / "depth" / anno_info["depth"]

            image_info = self.image_dict("")
            image_info.update(
                {
                    "original_image_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "cam_in": anno_info["cam_in"],
                    "depth_metric": 1000.,
                }
            )

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv
        import cv2
        import numpy as np
        import random

        num_choices = 4

        depth_map = cv2.imread(image_info["depth_path"], -1)
        depth_map = depth_map / image_info["depth_metric"]

        true_indices = np.where(depth_map > 0.)

        if len(true_indices[0]) > 0:
            # 随机选择一个为 True 的坐标
            random_index = np.random.randint(len(true_indices[0]))
            x = true_indices[0][random_index]
            y = true_indices[1][random_index]
            # print(f"随机找到的为True的坐标是 ({x}, {y})")
        else:
            print("depth_map 中没有为True的值。")
        
        width, height = image_info["width"], image_info["height"]


        cam_in = image_info["cam_in"]

        question = f"What is the depth (in meters) at the coordinates ({round(x / width, 3)}, {round(y / height, 3)}) in the figure? The camera intrinsic parameters are as follows, Focal Length: {cam_in[0]}, Principal Point: ({cam_in[1]}, {cam_in[2]}), Distortion Parameters: {cam_in[3]}"
        
        gt = depth_map[x, y]

        wrong_choices_list = []

        for i in range(num_choices - 1):
            wrong_choices_list.append(round(random.uniform(0, 2 * gt), 3))

        qa_json = {
            "num_wrong_choices": num_choices - 1,
            "gt": gt,
            "question": question,
            "wrong_choices_list": wrong_choices_list
        }
        
        
        qa_json = BaseDataset.post_process(qa_json)
        return qa_json



class nuscenes(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/depth_data/nuscenes",
        "sampling_num": 50,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info_list = mmcv.load(Path(self.image_path) / "annotations.json")

        for anno_info in anno_info_list:
            rgb_path = Path(self.image_path) / "rgb" / anno_info["rgb"]
            depth_path = Path(self.image_path) / "depth" / anno_info["depth"]

            image_info = self.image_dict("")
            image_info.update(
                {
                    "original_image_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "cam_in": anno_info["cam_in"],
                    "depth_metric": 256.,
                }
            )

            self.images_info.append(image_info)


class kitti(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/depth_data/kitti",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info_list = mmcv.load(Path(self.image_path) / "annotations.json")

        for anno_info in anno_info_list:
            rgb_path = Path(self.image_path) / "rgb" / anno_info["rgb"]
            depth_path = Path(self.image_path) / "depth" / anno_info["depth"]

            image_info = self.image_dict("")
            image_info.update(
                {
                    "original_image_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "cam_in": anno_info["cam_in"],
                    "depth_metric": 256.,
                }
            )

            self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv
        import cv2
        import numpy as np
        import random

        num_choices = 4

        depth_map = cv2.imread(image_info["depth_path"], -1)
        depth_map = depth_map / image_info["depth_metric"]

        true_indices = np.where(depth_map > 0.)

        if len(true_indices[0]) > 0:
            # 随机选择一个为 True 的坐标
            random_index = np.random.randint(len(true_indices[0]))
            x = true_indices[0][random_index]
            y = true_indices[1][random_index]
            # print(f"随机找到的为True的坐标是 ({x}, {y})")
        else:
            print("depth_map 中没有为True的值。")
        
        width, height = image_info["width"], image_info["height"]


        cam_in = image_info["cam_in"]

        question = f"What is the depth (in meters) at the coordinates ({round(x / width, 3)}, {round(y / height, 3)}) in the figure? The camera intrinsic parameters are as follows, Focal Length: {cam_in[0]}, Principal Point: ({cam_in[1]}, {cam_in[2]}), Distortion Parameters: {cam_in[3]}"
        
        gt = depth_map[x, y]

        wrong_choices_list = []

        for i in range(num_choices - 1):
            wrong_choices_list.append(round(random.uniform(0, 2 * gt), 3))

        qa_json = {
            "num_wrong_choices": num_choices - 1,
            "gt": gt,
            "question": question,
            "wrong_choices_list": wrong_choices_list
        }
        
        
        qa_json = BaseDataset.post_process(qa_json)
        return qa_json
