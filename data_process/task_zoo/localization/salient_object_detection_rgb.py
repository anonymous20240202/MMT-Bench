import numpy as np
import os
from base_dataset import BaseDataset
from PIL import Image
from collections import defaultdict
import cv2
from scipy import ndimage
from tqdm import tqdm

from prompt.utils import *

def count_connected_regions(mask):
    # 找到连通区域
    labeled, num_features = ndimage.label(mask)
    
    # 计算每个连通区域的面积
    region_sizes = np.bincount(labeled.flatten())

    # 设置小区域的标签为 0（不考虑）
    labeled[np.isin(labeled, np.where(region_sizes < 16))] = 0

    # 重新计算连通区域个数（排除小于 16 的区域）
    labeled[labeled > 0] = 1
    labeled, num_features = ndimage.label(labeled)
    
    return num_features

def mask2bbox(mask):
    # 获取所有非零元素的行索引和列索引
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # 找出最小和最大的行索引和列索引
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # 返回边界框坐标
    return int(x_min), int(y_min), int(x_max), int(y_max)

class MSRA10K(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/localization/salient_object_detection_rgb/MSRA10K_Imgs_GT/Imgs",
        "anno_path": "/path/to/TaskOnomy_Evaluation/localization/salient_object_detection_rgb/MSRA10K_Imgs_GT/Imgs",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "Note that this dataset have no category space and no train set (only used for testing). We use foreground as the category name"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = []
        file_list = os.listdir(self.DATA_METAINFO['image_path'])
        image_list = [x for x in file_list if x.endswith('.jpg')]
        if '.ipynb_checkpoints' in image_list:
            image_list.remove('.ipynb_checkpoints')
        for im in tqdm(image_list):
            original_image_path = os.path.join(self.DATA_METAINFO['image_path'], im)
            img = cv2.imread(original_image_path)
            h, w = img.shape[:2]
            mask_path = os.path.join(self.DATA_METAINFO['anno_path'], im.replace('.jpg', '.png'))
            gt_mask = np.asarray(Image.open(mask_path))
            gt_mask = gt_mask > 128
            num_region = count_connected_regions(gt_mask)
            if num_region > 1:
                #print('>>>{} has {} connected regions...'.format(original_image_path, num_region))
                continue
            x1, y1, x2, y2 = mask2bbox(gt_mask)
            
            classwise_boxes = defaultdict(list)
            classwise_boxes['foreground'].append([x1, y1, x2, y2])
            
            info = {
            'source': self.dataset_name,
            'category': 'No category defination. Use foreground as default.',#list(classwise_boxes.keys()),
            'boxes': classwise_boxes,
            'original_image_path': original_image_path.replace('/cpfs01/user/linyuqi/datasets', ' /path/to'),
            'height': h,
            'width': w
            }
            self.images_info.append(info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        width = image_info["width"]
        height = image_info["height"]
        question = f"Please detect the salient foreground object in this image and represent them using a single bounding box. Provide the output for the detected area in the format [x, y, w, h]. This format represents the bounding box, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as its width and height. Note that the width of the input image is {width} and the height is {height}."
        
        bbox = image_info["boxes"]['foreground'][0]
        bbox = xyxy2xywh(bbox)

        wrong_choices_list = generate_incorrect_bounding_box_from_single_bbox(bbox, width, height, num_choices-1)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_bounding_box_from_single_bbox(bbox, width, height, num_choices-1)
                    
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": bbox,
                    "question": question,
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json
            
            
class DUTS(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/localization/salient_object_detection_rgb/DUTS-TE/DUTS-TE-Image",
        "anno_path": "/path/to/TaskOnomy_Evaluation/localization/salient_object_detection_rgb/DUTS-TE/DUTS-TE-Mask",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "Note that this dataset have no category space. We use foreground as the category name"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        
    
    def parse_images_info(self):
        self.images_info = []
        image_list = os.listdir(self.DATA_METAINFO['image_path'])
        if '.ipynb_checkpoints' in image_list:
            image_list.remove('.ipynb_checkpoints')
        for im in tqdm(image_list):
            original_image_path = os.path.join(self.DATA_METAINFO['image_path'], im)
            img = cv2.imread(original_image_path)
            h, w = img.shape[:2]
            mask_path = os.path.join(self.DATA_METAINFO['anno_path'], im.replace('.jpg', '.png'))
            gt_mask = np.asarray(Image.open(mask_path))
            gt_mask = gt_mask > 128
            num_region = count_connected_regions(gt_mask)
            if num_region > 1:
                #print('>>>{} has {} connected regions...'.format(original_image_path, num_region))
                continue
            x1, y1, x2, y2 = mask2bbox(gt_mask)
            
            classwise_boxes = defaultdict(list)
            classwise_boxes['foreground'].append([x1, y1, x2, y2])
            
            
            info = {
            'source': self.dataset_name,
            'category': 'No category defination. Use foreground as default.',#list(classwise_boxes.keys()),
            'boxes': classwise_boxes,
            'original_image_path':original_image_path.replace('/cpfs01/user/linyuqi/datasets', ' /path/to'),
            'height': h,
            'width': w
            }
            self.images_info.append(info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        width = image_info["width"]
        height = image_info["height"]
        question = f"Please detect the salient foreground object in this image and represent them using a single bounding box. Provide the output for the detected area in the format [x, y, w, h]. This format represents the bounding box, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as its width and height. Note that the width of the input image is {width} and the height is {height}."
        
        bbox = image_info["boxes"]['foreground'][0]
        bbox = xyxy2xywh(bbox)

        wrong_choices_list = generate_incorrect_bounding_box_from_single_bbox(bbox, width, height, num_choices-1)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_bounding_box_from_single_bbox(bbox, width, height, num_choices-1)
                    
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": bbox,
                    "question": question,
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json