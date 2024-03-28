from pathlib import Path

from base_dataset import BaseDataset

from tqdm import tqdm

import numpy as np
import random
import mmcv
import os

import numpy as np
import cv2
from scipy import ndimage
from tqdm import tqdm
import numpy as np
import os
from base_dataset import BaseDataset
from PIL import Image
import matplotlib.pyplot as plt


import random


def generate_shifted_polygons(original_polygon, width, height, num_polygons):
    """
    Generates shifted versions of a given polygon.

    Args:
    original_polygon (numpy.ndarray): The original polygon vertices.
    width (int): The width of the image.
    height (int): The height of the image.
    num_polygons (int): Number of shifted polygons to generate.

    Returns:
    list: A list of numpy arrays, each representing a shifted polygon.
    """
    shifted_polygons = []

    for _ in range(num_polygons):
        shifted_polygon = []

        for point in original_polygon:
            # 随机偏移量，确保点仍在图片内
            shift_x = random.randint(-10, 10)
            shift_y = random.randint(-10, 10)

            new_x = min(max(point[0] + shift_x, 0), width - 1)
            new_y = min(max(point[1] + shift_y, 0), height - 1)

            shifted_polygon.append([new_x, new_y])

        shifted_polygons.append(shifted_polygon)

    return shifted_polygons



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


def mask_to_polygon_bool(mask):
    """
    Converts a boolean mask to a polygon representation.

    Args:
    mask (numpy.ndarray): A 2D numpy array representing the mask as boolean values.

    Returns:
    numpy.ndarray: An array of coordinates representing the polygon vertices.
    """
    # 将布尔掩码转换为uint8类型
    mask_uint8 = (mask * 255).astype(np.uint8)

    # 查找轮廓
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有找到轮廓，则返回空列表
    if not contours:
        return []

    # 选择最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 使用多边形近似
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx_poly = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 返回多边形的坐标点
    return approx_poly.reshape(-1, 2)


class davis_interactive(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/leijiayi/images/Interactive Segmentation/DAVIS",
        "anno_path": "/path/to/samples/interactive_segmentation/metadata_info_new.json",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = []
        anno_info = mmcv.load(self.anno_path)

        for image_info in anno_info["image"]:
            if image_info["source"] != "DAVIS":
                continue
            
            image_id = Path(image_info["original_image_path"]).stem
            original_image_path = os.path.join(self.image_path, f"images/{image_id}.jpg")

            mask_path = os.path.join(self.image_path, f"label/{image_id}.png")

            gt_mask = np.asarray(Image.open(mask_path))
            gt_mask = gt_mask > 0
            pass

            num_region = count_connected_regions(gt_mask)       
            if num_region > 1:
                #print('>>>{} has {} connected regions...'.format(original_image_path, num_region))
                continue
            x1, y1, x2, y2 = mask2bbox(gt_mask)

            polygon_point_list = mask_to_polygon_bool(gt_mask).tolist()

            image_info["source"] = self.dataset_name
            image_info["bbox"] = [x1, y1, x2, y2]

            image_info["polygon"] = polygon_point_list
            image_info["original_image_path"] = original_image_path

            self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import json
        import mmcv

        num_choices = 4
        width = image_info["width"]
        height = image_info["height"]

        # question, choice_list, gt_index, 
        question = f"I will give you a bounding box (marked as RED). Please locate the polygon points of the corresponding object in this image. Note that the width of the input image is given as {width} and the height as {height}."
        question_visual_mark = f"I will give you a bounding box (marked as RED). Please locate the polygon points of the corresponding object in this image. Note that the width of the input image is given as {width} and the height as {height}."

        polygon = image_info["polygon"]
        bbox = image_info["bbox"]

        wrong_choices_list = generate_shifted_polygons(polygon, width, height, 3)

        qa_json = {
            "num_wrong_choices": num_choices - 1,
            "gt": polygon,
            "question": question,
            "wrong_choices_list": wrong_choices_list
        }
        qa_json = BaseDataset.post_process(qa_json, question=question)

        BaseDataset.exist_or_mkdir(save_image_path)
        bbox_list = [np.array([bbox])]
        color_list = ["red"]
        new_image_path = str(Path(save_image_path) / BaseDataset.new_image_name())
        mmcv.imshow_bboxes(image_info["original_image_path"], bbox_list, show=False, out_file=new_image_path, thickness=2, colors=color_list)

        qa_json["marked_image_path"] = new_image_path

        return qa_json

class berkley_interactive(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/leijiayi/images/Interactive Segmentation/Berkeley",
        "anno_path": "/path/to/samples/interactive_segmentation/metadata_info_new.json",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = []
        anno_info = mmcv.load(self.anno_path)

        for image_info in anno_info["image"]:
            if image_info["source"] != "Berkley":
                continue
            
            image_id = Path(image_info["original_image_path"]).stem
            original_image_path = os.path.join(self.image_path, f"images/{image_id}.jpg")

            mask_path = os.path.join(self.image_path, f"label/{image_id}.png")

            gt_mask = np.asarray(Image.open(mask_path))
            gt_mask = gt_mask > 0
            gt_mask = gt_mask[:, :, 0]
            pass

            num_region = count_connected_regions(gt_mask)       
            if num_region > 1:
                #print('>>>{} has {} connected regions...'.format(original_image_path, num_region))
                continue
            x1, y1, x2, y2 = mask2bbox(gt_mask)

            polygon_point_list = mask_to_polygon_bool(gt_mask).tolist()

            image_info["source"] = self.dataset_name
            image_info["bbox"] = [x1, y1, x2, y2]

            image_info["polygon"] = polygon_point_list
            image_info["original_image_path"] = original_image_path

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import json
        import mmcv

        num_choices = 4
        width = image_info["width"]
        height = image_info["height"]

        # question, choice_list, gt_index, 
        question = f"I will give you a bounding box (marked as RED). Please locate the polygon points of the corresponding object in this image. Note that the width of the input image is given as {width} and the height as {height}."
        question_visual_mark = f"I will give you a bounding box (marked as RED). Please locate the polygon points of the corresponding object in this image. Note that the width of the input image is given as {width} and the height as {height}."

        polygon = image_info["polygon"]
        bbox = image_info["bbox"]

        wrong_choices_list = generate_shifted_polygons(polygon, width, height, 3)

        qa_json = {
            "num_wrong_choices": num_choices - 1,
            "gt": polygon,
            "question": question,
            "wrong_choices_list": wrong_choices_list
        }
        qa_json = BaseDataset.post_process(qa_json, question=question)

        BaseDataset.exist_or_mkdir(save_image_path)
        bbox_list = [np.array([bbox])]
        color_list = ["red"]
        new_image_path = str(Path(save_image_path) / BaseDataset.new_image_name())
        mmcv.imshow_bboxes(image_info["original_image_path"], bbox_list, show=False, out_file=new_image_path, thickness=2, colors=color_list)

        qa_json["marked_image_path"] = new_image_path

        return qa_json
