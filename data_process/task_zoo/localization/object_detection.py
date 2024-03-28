import json
import numpy as np
import os
from collections import defaultdict
import uuid
from tqdm import tqdm
from base_dataset import BaseDataset
import cv2
from lxml import etree

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

# All coco categories, together with their nice-looking visualization colors
# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json
COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
    {"color": [255, 255, 128], "isthing": 0, "id": 92, "name": "banner"},
    {"color": [147, 211, 203], "isthing": 0, "id": 93, "name": "blanket"},
    {"color": [150, 100, 100], "isthing": 0, "id": 95, "name": "bridge"},
    {"color": [168, 171, 172], "isthing": 0, "id": 100, "name": "cardboard"},
    {"color": [146, 112, 198], "isthing": 0, "id": 107, "name": "counter"},
    {"color": [210, 170, 100], "isthing": 0, "id": 109, "name": "curtain"},
    {"color": [92, 136, 89], "isthing": 0, "id": 112, "name": "door-stuff"},
    {"color": [218, 88, 184], "isthing": 0, "id": 118, "name": "floor-wood"},
    {"color": [241, 129, 0], "isthing": 0, "id": 119, "name": "flower"},
    {"color": [217, 17, 255], "isthing": 0, "id": 122, "name": "fruit"},
    {"color": [124, 74, 181], "isthing": 0, "id": 125, "name": "gravel"},
    {"color": [70, 70, 70], "isthing": 0, "id": 128, "name": "house"},
    {"color": [255, 228, 255], "isthing": 0, "id": 130, "name": "light"},
    {"color": [154, 208, 0], "isthing": 0, "id": 133, "name": "mirror-stuff"},
    {"color": [193, 0, 92], "isthing": 0, "id": 138, "name": "net"},
    {"color": [76, 91, 113], "isthing": 0, "id": 141, "name": "pillow"},
    {"color": [255, 180, 195], "isthing": 0, "id": 144, "name": "platform"},
    {"color": [106, 154, 176], "isthing": 0, "id": 145, "name": "playingfield"},
    {"color": [230, 150, 140], "isthing": 0, "id": 147, "name": "railroad"},
    {"color": [60, 143, 255], "isthing": 0, "id": 148, "name": "river"},
    {"color": [128, 64, 128], "isthing": 0, "id": 149, "name": "road"},
    {"color": [92, 82, 55], "isthing": 0, "id": 151, "name": "roof"},
    {"color": [254, 212, 124], "isthing": 0, "id": 154, "name": "sand"},
    {"color": [73, 77, 174], "isthing": 0, "id": 155, "name": "sea"},
    {"color": [255, 160, 98], "isthing": 0, "id": 156, "name": "shelf"},
    {"color": [255, 255, 255], "isthing": 0, "id": 159, "name": "snow"},
    {"color": [104, 84, 109], "isthing": 0, "id": 161, "name": "stairs"},
    {"color": [169, 164, 131], "isthing": 0, "id": 166, "name": "tent"},
    {"color": [225, 199, 255], "isthing": 0, "id": 168, "name": "towel"},
    {"color": [137, 54, 74], "isthing": 0, "id": 171, "name": "wall-brick"},
    {"color": [135, 158, 223], "isthing": 0, "id": 175, "name": "wall-stone"},
    {"color": [7, 246, 231], "isthing": 0, "id": 176, "name": "wall-tile"},
    {"color": [107, 255, 200], "isthing": 0, "id": 177, "name": "wall-wood"},
    {"color": [58, 41, 149], "isthing": 0, "id": 178, "name": "water-other"},
    {"color": [183, 121, 142], "isthing": 0, "id": 180, "name": "window-blind"},
    {"color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window-other"},
    {"color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree-merged"},
    {"color": [190, 153, 153], "isthing": 0, "id": 185, "name": "fence-merged"},
    {"color": [146, 139, 141], "isthing": 0, "id": 186, "name": "ceiling-merged"},
    {"color": [70, 130, 180], "isthing": 0, "id": 187, "name": "sky-other-merged"},
    {"color": [134, 199, 156], "isthing": 0, "id": 188, "name": "cabinet-merged"},
    {"color": [209, 226, 140], "isthing": 0, "id": 189, "name": "table-merged"},
    {"color": [96, 36, 108], "isthing": 0, "id": 190, "name": "floor-other-merged"},
    {"color": [96, 96, 96], "isthing": 0, "id": 191, "name": "pavement-merged"},
    {"color": [64, 170, 64], "isthing": 0, "id": 192, "name": "mountain-merged"},
    {"color": [152, 251, 152], "isthing": 0, "id": 193, "name": "grass-merged"},
    {"color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt-merged"},
    {"color": [206, 186, 171], "isthing": 0, "id": 195, "name": "paper-merged"},
    {"color": [152, 161, 64], "isthing": 0, "id": 196, "name": "food-other-merged"},
    {"color": [116, 112, 0], "isthing": 0, "id": 197, "name": "building-other-merged"},
    {"color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock-merged"},
    {"color": [102, 102, 156], "isthing": 0, "id": 199, "name": "wall-other-merged"},
    {"color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug-merged"},
]

COCO_CLASSID_2_CLASSNAME = {}
category80 = []
for cate in COCO_CATEGORIES:
    COCO_CLASSID_2_CLASSNAME[cate['id']] = cate['name']
    if cate['id'] <= 90:
        category80.append(cate['name'])
    
#print(len(category80))

class coco_det(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/localization/object_detection/coco/images/val2017",
        "anno_path": "/path/to/TaskOnomy_Evaluation/localization/object_detection/coco/annotations",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.category_space = category80
        self.dataset_info["category_space"] = self.category_space
    
    def parse_images_info(self):
        self.images_info = list()

        anno_path = os.path.join(self.DATA_METAINFO['anno_path'], 'instances_val2017.json')
        with open(anno_path, "r") as f:
            data = json.load(f)
        
        imid2name = {}
        for im in data['images']:
            #print(im.keys())
            imid2name[im['id']] = im['file_name']

        pred_by_image = defaultdict(list)
        for p in data['annotations']:
            pred_by_image[p["image_id"]].append(p)

        images = list(pred_by_image.keys())
        print(len(images))
        
        for im in tqdm(images):
            annos = pred_by_image[im]
            ori_name = imid2name[im]
            original_image_path = os.path.join(self.DATA_METAINFO['image_path'], ori_name)
            img = cv2.imread(original_image_path)
            h, w = img.shape[:2]
            
            classwise_boxes = defaultdict(list)
            for anno in annos:
                box_ = anno['bbox']
                box_[2] += box_[0]
                box_[3] += box_[1]
                box_ = [round(b, 2) for b in box_]
                classwise_boxes[COCO_CLASSID_2_CLASSNAME[anno['category_id']]].append(box_)
        

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
            
            
class VOC2012_det(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/localization/object_detection/VOC2012/JPEGImages",
        "anno_path": "/path/to/TaskOnomy_Evaluation/localization/object_detection/VOC2012/Annotations",
        "split_path": "/path/to/TaskOnomy_Evaluation/localization/object_detection/VOC2012/ImageSets/Main",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "category_space": ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.dataset_info["category_space"] = self.DATA_METAINFO['category_space']
    
    def parse_images_info(self):
        self.images_info = []
        #image_list = os.listdir(self.DATA_METAINFO['image_path'])
        split_file = os.path.join(self.DATA_METAINFO['split_path'], 'val.txt')
        image_list = list(np.loadtxt(split_file, dtype=str))
        image_list = [x+'.jpg' for x in image_list]
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
