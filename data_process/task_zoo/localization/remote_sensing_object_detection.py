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


class DIOR(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/localization/remote_sensing_object_detection/DIOR/JPEGImages-test",
        "anno_path": "/path/to/TaskOnomy_Evaluation/localization/remote_sensing_object_detection/DIOR/Annotations/Horizontal_Bounding_Boxes",
        "sampling_num": 190,
        "visual_input_component": ["remote_sensing_image"],
        "category_space": ['airplane', 'airport', 'baseball field', 'basketball court', 'bridge', 'chimney', 'dam', 'expressway service area', 'expressway toll station', 'harbor', 'golf course', 'ground track field', 'overpass', 'ship', 'stadium', 'storage tank', 'tennis court', 'train station', 'vehicle', 'wind mill'],
        "dataset_description": "XXX"
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
            xml = etree.fromstring(xml_str.encode('utf-8'))  # etree包 读取xml文件
            data = parse_xml_to_dict(xml)["annotation"]

            width = int(data['size']['width'])
            height = int(data['size']['height'])
            assert h == height and w == width
            
            classwise_boxes = defaultdict(list)
            for obj in data["object"]:
                name = obj["name"]
                x1, y1, x2, y2 = obj["bndbox"]["xmin"], obj["bndbox"]["ymin"], obj["bndbox"]["xmax"], obj["bndbox"]["ymax"]
                classwise_boxes[name].append([x1, y1, x2, y2])
            
            len_num = sum([len(_) for _ in classwise_boxes.values()])
            if len_num > 6:
                continue
            
            info = {
            'source': self.dataset_name,
            'category': list(classwise_boxes.keys()),
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
        question = f"Please detect all instances of the following categories in this image: {', '.join(dataset_info['category_space'])}. For each detected object, provide the output in the format category:[x, y, w, h]. This format represents the bounding box for each object, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as the width and height of the bounding box. Note that the width of the input image is {width} and the height is {height}."
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))
        bbox_list = []
        label_list = []
        for cate, value in image_info["boxes"].items():
            label_list.extend([cate for _ in range(len(value))])
            bbox_list.extend(value)
        
        bbox_list = [xyxy2xywh(bbox) for bbox in bbox_list]


        _t = []
        for bbox, cate in zip(bbox_list, label_list):
            _t.append(f"{cate}: {bbox}")
        gt = ", ".join(_t)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_bounding_boxes_with_labels(bbox_list, label_list, image_info['width'], image_info['height'], num_choices - 1)
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

            

class VisDrone(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/localization/remote_sensing_object_detection/VisDrone/VisDrone2019-DET-val/images",
        "anno_path": "/path/to/TaskOnomy_Evaluation/localization/remote_sensing_object_detection/VisDrone/VisDrone2019-DET-val/annotations",
        "sampling_num": 100,
        "visual_input_component": ["remote_sensing_image"],
        "category_space": ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus','motor'],
        "dataset_description": "xxx"
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
            
            label_path = os.path.join(self.DATA_METAINFO['anno_path'], im.replace('.jpg', '.txt'))
            annos = np.loadtxt(label_path, delimiter=',').reshape(-1, 8)
            
            cateid2name = {}
            for i in range(1, 11):
                cateid2name[i] = self.DATA_METAINFO['category_space'][i-1]
            
            classwise_boxes = defaultdict(list)
            #<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
            for ann in annos:
                x1, y1, w0, h0, score, cateid, trun, occ = ann
                x2 = x1 + w0
                y2 = y1 + h0
                if score == 1:
                    classwise_boxes[cateid2name[cateid]].append([int(x1), int(y1), int(x2), int(y2)])
            
            len_num = sum([len(_) for _ in classwise_boxes.values()])
            if len_num > 4:
                continue
            info = {
            'source': self.dataset_name,
            'category': list(classwise_boxes.keys()),
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
        question = f"Please detect all instances of the following categories in this image: {', '.join(dataset_info['category_space'])}. For each detected object, provide the output in the format category:[x, y, w, h]. This format represents the bounding box for each object, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as the width and height of the bounding box. Note that the width of the input image is {width} and the height is {height}."
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))
        bbox_list = []
        label_list = []
        for cate, value in image_info["boxes"].items():
            label_list.extend([cate for _ in range(len(value))])
            bbox_list.extend(value)
        
        bbox_list = [xyxy2xywh(bbox) for bbox in bbox_list]


        _t = []
        for bbox, cate in zip(bbox_list, label_list):
            _t.append(f"{cate}: {bbox}")
        gt = ", ".join(_t)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_bounding_boxes_with_labels(bbox_list, label_list, image_info['width'], image_info['height'], num_choices - 1)
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