import numpy as np
import os
from base_dataset import BaseDataset
from PIL import Image
from collections import defaultdict
import cv2
from scipy import ndimage
from lxml import etree
from tqdm import tqdm

from prompt.utils import *

def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

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

class Trans10K(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/localization/transparent_object_detection/Trans10K/validation/easy/images",
        "anno_path": "/path/to/TaskOnomy_Evaluation/localization/transparent_object_detection/Trans10K/validation/easy/masks",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "Note that this dataset have no category space. We use foreground as the category name"
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
            mask_path = os.path.join(self.DATA_METAINFO['anno_path'], im.replace('.jpg', '_mask.png'))
            gt_mask = np.asarray(Image.open(mask_path))[:, :, 0]
            gt_mask = gt_mask > 0
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
        question = f"Please detect all transparent foreground instances in this image. For each detected object, provide the output in the format [x, y, w, h]. This format represents the bounding box for each object, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as the width and height of the bounding box. Note that the width of the input image is {width} and the height is {height}."
        bbox_list = []
        label_list = []
        for cate, value in image_info["boxes"].items():
            label_list.extend([cate for _ in range(len(value))])
            bbox_list.extend(value)
        
        bbox_list = [xyxy2xywh(bbox) for bbox in bbox_list]


        _t = []
        for bbox, cate in zip(bbox_list, label_list):
            _t.append(f"{cate}: {bbox}")
        gt = ", ".join([f"{bbox}" for bbox in bbox_list])

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_bounding_boxes_with_labels(bbox_list, label_list, image_info['width'], image_info['height'], num_choices - 1)
                new_wrong_choices_list = []
                for wrong_choice in wrong_choices_list:
                    _t = []
                    bbox_list = wrong_choice[0]
                    # for bbox, cate in zip(wrong_choice[0], wrong_choice[1]):
                    #     _t.append(f"{cate}: {bbox}")
                    
                    new_wrong_choices_list.append(", ".join([f"{bbox}" for bbox in bbox_list]))
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
            
            
class Transparent_Object_Images(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/localization/transparent_object_detection/Transparent_Object_Images/transparent_glass/transparent_glass",
        "anno_path": "/path/to/TaskOnomy_Evaluation/localization/transparent_object_detection/Transparent_Object_Images/annotations/annotations",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "category_space": ["transparent_glass"],
        "dataset_description": "Note that this dataset is designed to detect transparent_glass."
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.dataset_info["category_space"] = self.DATA_METAINFO['category_space']
    
    def parse_images_info(self):
        self.images_info = []
        image_list = os.listdir(self.DATA_METAINFO['image_path'])
        if '.ipynb_checkpoints' in image_list:
            image_list.remove('.ipynb_checkpoints')
        for im in tqdm(image_list):
            original_image_path = os.path.join(self.DATA_METAINFO['image_path'], im)
            img = cv2.imread(original_image_path)
            h, w = img.shape[:2]
            
            xmlfile = os.path.join(self.DATA_METAINFO['anno_path'], im.replace('.jpg', '.xml'))
            with open(xmlfile) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)  # etree包 读取xml文件
            data = parse_xml_to_dict(xml)["annotation"]

            width = int(data['size']['width'])
            height = int(data['size']['height'])
            assert h == height and w == width
            
            classwise_boxes = defaultdict(list)
            for obj in data["object"]:
                x1, y1, x2, y2 = obj["bndbox"]["xmin"], obj["bndbox"]["ymin"], obj["bndbox"]["xmax"], obj["bndbox"]["ymax"]
                classwise_boxes['transparent_glass'].append([x1, y1, x2, y2])
            
            
            info = {
            'source': self.dataset_name,
            'category': 'transparent_glass',#list(classwise_boxes.keys()),
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
        question = f"Please detect all transparent foreground instances in this image. For each detected object, provide the output in the format [x, y, w, h]. This format represents the bounding box for each object, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as the width and height of the bounding box. Note that the width of the input image is {width} and the height is {height}."
        bbox_list = []
        label_list = []
        for cate, value in image_info["boxes"].items():
            label_list.extend([cate for _ in range(len(value))])
            bbox_list.extend(value)
        
        bbox_list = [xyxy2xywh(bbox) for bbox in bbox_list]


        _t = []
        for bbox, cate in zip(bbox_list, label_list):
            _t.append(f"{cate}: {bbox}")
        gt = ", ".join([f"{bbox}" for bbox in bbox_list])

        i = 0
        while i <= 100:
            try:
                wrong_choices_list = generate_incorrect_bounding_boxes_with_labels(bbox_list, label_list, image_info['width'], image_info['height'], num_choices - 1)
                new_wrong_choices_list = []
                for wrong_choice in wrong_choices_list:
                    _t = []
                    bbox_list = wrong_choice[0]
                    
                    new_wrong_choices_list.append(", ".join([f"{bbox}" for bbox in bbox_list]))
                _qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt,
                    "question": question,
                    "wrong_choices_list": new_wrong_choices_list
                }
                qa_json = BaseDataset.post_process(_qa_json, question=question)
                break
            except Exception as e:
                print(e)
                i += 1

        return qa_json
            