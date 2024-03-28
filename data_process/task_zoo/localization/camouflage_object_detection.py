import numpy as np
import os
from base_dataset import BaseDataset
from PIL import Image
from collections import defaultdict
import cv2
from tqdm import tqdm

from prompt.utils import *

def mask2bbox(mask):
    # 获取所有非零元素的行索引和列索引
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # 找出最小和最大的行索引和列索引
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # 返回边界框坐标
    return int(x_min), int(y_min), int(x_max), int(y_max)


class COD10K(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/localization/camouflage_object_detection/COD10K-v3/Test/Image",
        "anno_path": "/path/to/TaskOnomy_Evaluation/localization/camouflage_object_detection/COD10K-v3/Test/GT_Instance",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    Super_Class_Dictionary = {'1':'Aquatic', '2':'Terrestrial', '3':'Flying', '4':'Amphibian', '5':'Other'}
    Sub_Class_Dictionary = {'1':'batFish','2':'clownFish','3':'crab','4':'crocodile','5':'crocodileFish','6':'fish','7':'flounder','8':'frogFish','9':'ghostPipefish','10':'leafySeaDragon','11':'octopus','12':'pagurian','13':'pipefish','14':'scorpionFish','15':'seaHorse','16':'shrimp','17':'slug','18':'starFish','19':'stingaree','20':'turtle','21':'ant','22':'bug','23':'cat','24':'caterpillar','25':'centipede','26':'chameleon','27':'cheetah','28':'deer','29':'dog','30':'duck','31':'gecko','32':'giraffe','33':'grouse','34':'human','35':'kangaroo','36':'leopard','37':'lion','38':'lizard','39':'monkey','40':'rabbit','41':'reccoon','42':'sciuridae','43':'sheep','44':'snake','45':'spider','46':'stickInsect','47':'tiger','48':'wolf','49':'worm','50':'bat','51':'bee','52':'beetle','53':'bird','54':'bittern','55':'butterfly','56':'cicada','57':'dragonfly','58':'frogmouth','59':'grasshopper','60':'heron','61':'katydid','62':'mantis','63':'mockingbird','64':'moth','65':'owl','66':'owlfly','67':'frog','68':'toad','69':'other'}
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.super_category_space = self.Super_Class_Dictionary
        self.dataset_info["super_category_space"] = self.super_category_space
        self.category_space = self.Sub_Class_Dictionary
        self.dataset_info["category_space"] = self.category_space
    
    def parse_images_info(self):
        self.images_info = []
        image_list = os.listdir(self.DATA_METAINFO['image_path'])
        for im in tqdm(image_list):
            #COD10K-CAM-SuperNumber-SuperClass-SubNumber-SubClass-ImageNumber
            if im.split('-')[1] == 'NonCAM':
                continue
            _, cam, _, superclass, _, subclass, _ = im.split('-')
            
            original_image_path = os.path.join(self.DATA_METAINFO['image_path'], im)
            img = cv2.imread(original_image_path)
            h, w = img.shape[:2]
            mask_path = os.path.join(self.DATA_METAINFO['anno_path'], im.replace('.jpg', '.png'))
            gt_mask = np.asarray(Image.open(mask_path))
            instances = list(np.unique(gt_mask))
            
            classwise_boxes = defaultdict(list)
            for ins in instances:
                if ins == 0:
                    continue
                ins_mask = gt_mask==ins
                x1, y1, x2, y2 = mask2bbox(ins_mask)
                classwise_boxes[subclass].append([x1, y1, x2, y2])
            
            
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
            
            
class NC4K(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/localization/camouflage_object_detection/NC4K/Imgs",
        "anno_path": "/path/to/TaskOnomy_Evaluation/localization/camouflage_object_detection/NC4K/Instance",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "Note that this dataset have no category space and no train set (only used for testing). We use foreground as the category name"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        
    
    def parse_images_info(self):
        self.images_info = []
        image_list = os.listdir(self.DATA_METAINFO['image_path'])
        for im in tqdm(image_list):
            original_image_path = os.path.join(self.DATA_METAINFO['image_path'], im)
            img = cv2.imread(original_image_path)
            h, w = img.shape[:2]
            mask_path = os.path.join(self.DATA_METAINFO['anno_path'], im.replace('.jpg', '.png'))
            gt_mask = np.asarray(Image.open(mask_path))
            instances = list(np.unique(gt_mask))
            
            classwise_boxes = defaultdict(list)
            for ins in instances:
                if ins == 0:
                    continue
                ins_mask = gt_mask==ins
                x1, y1, x2, y2 = mask2bbox(ins_mask)
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
        question = f"Please detect all camouflaged foreground instances in this image. For each detected object, provide the output in the format [x, y, w, h]. This format represents the bounding box for each object, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as the width and height of the bounding box. Note that the width of the input image is {width} and the height is {height}."
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
            