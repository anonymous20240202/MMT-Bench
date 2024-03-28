from pathlib import Path

import mmcv

from base_dataset import BaseDataset

from prompt.utils import *


class dota(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/rotate/dota/images/images",
        "anno_path": "/path/to/lvlm_evaluation/data/rotate/dota/labelTxt-v1.5",
        "sampling_num": 30,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for image_path in Path(self.image_path).iterdir():
            anno_path = Path(self.anno_path) / f"{image_path.stem}.txt"
            anno_info_list = mmcv.list_from_file(anno_path)

            bbox_list = []
            category_list = []
            for instance_anno_info in anno_info_list[2:]:
                x1, y1, x2, y2, x3, y3, x4, y4, category, difficult = instance_anno_info.split()
                bbox_list.append([x1, y1, x2, y2, x3, y3, x4, y4])
                category_list.append(category)

                self.category_space.append(category)

            width, height = self.get_image_width_height(image_path)

            if len(bbox_list) >= 4:
                continue
            
            image_info = self.image_dict("")
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "bounding_box_coordinates": bbox_list,
                    "category": category_list
                }
            )

            self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = list(set(self.category_space))

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        width = image_info["width"]
        height = image_info["height"]
        question = f"Please localize all instances of the following categories in this image: {', '.join(dataset_info['category_space'])}. For each detected object, provide the output in the format: category: [x1, y1, x2, y2, ..., x24, y24], where category denotes the name of this object and [x1, y1, x2, y2, ..., x24, y24] stand for the coordinates of boundary points. Note that the width of the input image is given as {width} and the height as {height}."

        bbox_list = []
        label_list = []
        for cate, bbox in zip(image_info["category"], image_info["bounding_box_coordinates"]):
            label_list.append(cate)
            bbox_list.append([int(float(i)) for i in bbox])
        
        _t = []
        for bbox, cate in zip(bbox_list, label_list):
            _t.append(f"{cate}: {bbox}")
        gt = ", ".join(_t)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_rotated_bounding_boxes_with_labels(bbox_list, label_list, width, height, num_choices - 1)
                new_wrong_choices_list = []
                for wrong_choice in wrong_choices_list:
                    _t = []
                    for bbox, cate in zip(wrong_choice[0], wrong_choice[1]):
                        _t.append(f"{cate}: {bbox}")
                    
                    new_wrong_choices_list.append(", ".join(_t))
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt,
                    "question": question,
                    "wrong_choices_list": new_wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json


class ssdd_inshore(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/rotate/ssdd/test/inshore/images",
        "anno_path": "/path/to/lvlm_evaluation/data/rotate/ssdd/test/inshore/labelTxt",
        "sampling_num": 30,
        "visual_input_component": ["sar_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for image_path in Path(self.image_path).iterdir():
            anno_path = Path(self.anno_path) / f"{image_path.stem}.txt"
            anno_info_list = mmcv.list_from_file(anno_path)

            bbox_list = []
            category_list = []
            for instance_anno_info in anno_info_list:
                x1, y1, x2, y2, x3, y3, x4, y4, category, difficult = instance_anno_info.split()
                bbox_list.append([x1, y1, x2, y2, x3, y3, x4, y4])
                category_list.append(category)

                self.category_space.append(category)

            width, height = self.get_image_width_height(image_path)
            
            image_info = self.image_dict("")
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "bounding_box_coordinates": bbox_list,
                    "category": category_list
                }
            )

            self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = list(set(self.category_space))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        width = image_info["width"]
        height = image_info["height"]
        question = f"Please detect all ships in this SAR image. For each detected object, provide the output in the format: [x1, y1, x2, y2, x3, y3, x4, y4]. This format represents a rotated bounding box for each object. The points [x1, y1], [x2, y2], [x3, y3], and [x4, y4] are the coordinates of the four corners of the bounding box. Note that the width of the input image is given as {width} and the height as {height}."

        bbox_list = []
        label_list = []
        for cate, bbox in zip(image_info["category"], image_info["bounding_box_coordinates"]):
            label_list.append(cate)
            bbox_list.append([int(float(i)) for i in bbox])
        
        _t = []
        for bbox, cate in zip(bbox_list, label_list):
            _t.append(f"{bbox}")
        gt = ", ".join(_t)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_rotated_bounding_boxes_with_labels(bbox_list, label_list, width, height, num_choices - 1)
                new_wrong_choices_list = []
                for wrong_choice in wrong_choices_list:
                    _t = []
                    for bbox, cate in zip(wrong_choice[0], wrong_choice[1]):
                        _t.append(f"{bbox}")
                    
                    new_wrong_choices_list.append(", ".join(_t))
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt,
                    "question": question,
                    "wrong_choices_list": new_wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json


class ssdd_offshore(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/rotate/ssdd/test/offshore/images",
        "anno_path": "/path/to/lvlm_evaluation/data/rotate/ssdd/test/offshore/labelTxt",
        "sampling_num": 30,
        "visual_input_component": ["sar_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for image_path in Path(self.image_path).iterdir():
            anno_path = Path(self.anno_path) / f"{image_path.stem}.txt"
            anno_info_list = mmcv.list_from_file(anno_path)

            bbox_list = []
            category_list = []
            for instance_anno_info in anno_info_list:
                x1, y1, x2, y2, x3, y3, x4, y4, category, difficult = instance_anno_info.split()
                bbox_list.append([x1, y1, x2, y2, x3, y3, x4, y4])
                category_list.append(category)

                self.category_space.append(category)

            width, height = self.get_image_width_height(image_path)
            
            image_info = self.image_dict("")
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "bounding_box_coordinates": bbox_list,
                    "category": category_list
                }
            )

            self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = list(set(self.category_space))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        width = image_info["width"]
        height = image_info["height"]
        question = f"Please detect all ships in this SAR image. For each detected object, provide the output in the format: [x1, y1, x2, y2, x3, y3, x4, y4]. This format represents a rotated bounding box for each object. The points [x1, y1], [x2, y2], [x3, y3], and [x4, y4] are the coordinates of the four corners of the bounding box. Note that the width of the input image is given as {width} and the height as {height}."

        bbox_list = []
        label_list = []
        for cate, bbox in zip(image_info["category"], image_info["bounding_box_coordinates"]):
            label_list.append(cate)
            bbox_list.append([int(float(i)) for i in bbox])
        
        _t = []
        for bbox, cate in zip(bbox_list, label_list):
            _t.append(f"{bbox}")
        gt = ", ".join(_t)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_rotated_bounding_boxes_with_labels(bbox_list, label_list, width, height, num_choices - 1)
                new_wrong_choices_list = []
                for wrong_choice in wrong_choices_list:
                    _t = []
                    for bbox, cate in zip(wrong_choice[0], wrong_choice[1]):
                        _t.append(f"{bbox}")
                    
                    new_wrong_choices_list.append(", ".join(_t))
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt,
                    "question": question,
                    "wrong_choices_list": new_wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json
    