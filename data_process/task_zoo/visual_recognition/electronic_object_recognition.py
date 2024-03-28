from pathlib import Path
import csv
import os
import mmcv
from PIL import Image
import struct
import numpy as np

from base_dataset import BaseDataset
from lxml import etree
from tqdm import tqdm
from prompt.utils import *


def select_elements(lst, n1=0.6, n2=0.25, n3=0.15):
    if not lst:
        return []

    # 确保概率总和为 1
    if n1 + n2 + n3 != 1:
        raise ValueError("The sum of probabilities must be 1")

    # 选择要抽取的元素数量
    num_elements = random.choices([1, 2, 3], weights=[n1, n2, n3], k=1)[0]

    # 如果列表长度小于所需的元素数量，则返回整个列表
    if len(lst) < num_elements:
        return lst

    # 从列表中随机选择所需数量的元素
    return random.sample(lst, num_elements)


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



class e_waste(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/visual_recognition/electronic_object_recognition/e-waste/test",
        "sampling_num": 150,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        
        cates = os.listdir(self.DATA_METAINFO['image_path'])
        if '.ipynb_checkpoints' in cates:
            cates.remove('.ipynb_checkpoints')
        
        for cate in tqdm(cates):
            
            im_dir = os.path.join(self.DATA_METAINFO['image_path'], cate)
            im_list = os.listdir(im_dir)
            cate = cate.replace("_", " ")
            self.category_space.append(cate)
            
            for im in im_list:
                original_image_path = os.path.join(im_dir, im)
                info = {
                    'source': self.dataset_name,
                    'category': cate,
                    'original_image_path': original_image_path.replace('/cpfs01/user/linyuqi/datasets', ' /path/to'),
                    }
                self.images_info.append(info)
        self.dataset_info["category_space"] = self.category_space
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What category of the electronic object is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": [replace_underscore_with_space(i) for i in dataset_info["category_space"]],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "microwave",
                "question": question,
                "wrong_choices_list": ["mobile", "keyboard", "mouse"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": replace_underscore_with_space(image_info["category"]),
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)

        i = 0
        while i <= 10:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json


class electronics_object_image_dataset_computer_parts(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/visual_recognition/electronic_object_recognition/electronics_object_image_dataset_computer_parts/samples_for_clients/samples_for_clients",
        "anno_path": "/path/to/TaskOnomy_Evaluation/visual_recognition/electronic_object_recognition/electronics_object_image_dataset_computer_parts/annotations/annotations",
        "sampling_num": 50,
        "visual_input_component": ["natural_image"],
        "dataset_description": "This dataset have no val and test set. The total number of samples is 99 and there are some images containing multiple categories."
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        
        anno_list = os.listdir(self.DATA_METAINFO['anno_path'])
        if '.ipynb_checkpoints' in anno_list:
            anno_list.remove('.ipynb_checkpoints')
        for anno in tqdm(anno_list):
            xmlfile = os.path.join(self.DATA_METAINFO['anno_path'], anno)
            original_image_path =os.path.join(self.DATA_METAINFO['image_path'], anno.replace('.xml', '.jpg'))
            assert os.path.isfile(original_image_path), original_image_path
            
            with open(xmlfile) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str.encode('utf-8'))  # etree包 读取xml文件
            data = parse_xml_to_dict(xml)["annotation"]
            

            if "object" not in data.keys():
                continue 
            #if "object" in data.keys() and len(data["object"]) > 1:
            #    continue
            
            img_cates = []
            for obj in data["object"]:
                name = obj["name"]
                img_cates.append(name)
                self.category_space.append(name)
            info = {
                'source': self.dataset_name,
                'category': img_cates,
                'original_image_path': original_image_path.replace('/cpfs01/user/linyuqi/datasets', ' /path/to'),
                }
            self.images_info.append(info)
        self.category_space = list(set(self.category_space))
        self.dataset_info["category_space"] = self.category_space
        #print(len(self.images_info))

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What category of the electronic object is shown in the picture?"
        i = 0
        while i <= 10:
            try:
                gt = image_info["category"]
                wrong_chocies_list = [select_elements([i for i in dataset_info["category_space"]]) for i in range(num_choices - 1)]
                assert check_overlap_and_uniqueness(gt, wrong_chocies_list)
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": ",".join(image_info["category"]),
                    "question": question,
                    "wrong_choices_list": [",".join(list(i)) for i in wrong_chocies_list]
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except Exception as e:
                print("An error occurred:", e)

        return qa_json

def check_overlap_and_uniqueness(a, b):
    # 将所有子列表中的元素合并到一个集合中，以检查b中是否有重复元素
    all_elements_in_b = list()
    for sublist in b:
        if set(sublist) in all_elements_in_b:
            return False
        all_elements_in_b.append(set(sublist))

    if set(a) in all_elements_in_b:
        return False
    return True
