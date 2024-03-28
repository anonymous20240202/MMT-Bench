import random
from pathlib import Path
from collections import defaultdict

from base_dataset import BaseDataset

from tqdm import tqdm
import mmcv
import random
import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_and_save_five_images(image_paths, labels, save_path, dpi):
    """
    Plots five images side by side with their respective labels and saves the plot to a specified path.

    Parameters:
    image_paths (list of str): List of file paths for the images.
    labels (list of str): List of labels for the images.
    save_path (str): Path to save the combined image plot.
    dpi (int): Resolution of the saved image.
    """

    num = len(image_paths)
    if len(image_paths) != num or len(labels) != num:
        raise ValueError("Five image paths and labels are required.")

    # 读取图片
    # imgs = [mpimg.imread(img_path) for img_path in image_paths]
        # 读取图片
    imgs = []
    for image_path in image_paths:
        _img = mpimg.imread(image_path)

        if len(_img.shape) == 3 and _img.shape[2] == 3:
            pass
        else:
            _img = np.stack((_img,)*3, axis=-1)
        
        imgs.append(_img)

    # 创建图形和子图
    fig, axes = plt.subplots(1, num, figsize=(15, num))

    # 在子图上显示图片和标签
    for ax, img, label in zip(axes, imgs, labels):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')  # 关闭坐标轴

    # 调整子图之间的间隔
    plt.subplots_adjust(wspace=0.5)

    # 保存图形到指定路径
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)


class quickdraw_retrieval(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/sketch_retrieval/meta_info.json",
        "image_path": "/path/to/sketch_retrieval/",
        "sampling_num": 100,
        "visual_input_component": ["natural_image", "sketch_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        anno_info = mmcv.load(self.anno_path)

        self.images_info = []
        for image_info in anno_info["images"]:
            if image_info["source"] != "QuickDraw":
                continue
            

            query_image_path = image_info["image_query"].replace("/home/wenbo/Documents/data/LLM_eval_dataset/sketch_retrieval/", self.image_path)
            matched_image_path = image_info["image_candidate1"].replace("/home/wenbo/Documents/data/LLM_eval_dataset/sketch_retrieval/", self.image_path)
            unmatched_image_path_lsit = [
                image_info["image_candidate2"].replace("/home/wenbo/Documents/data/LLM_eval_dataset/sketch_retrieval/", self.image_path),
                image_info["image_candidate3"].replace("/home/wenbo/Documents/data/LLM_eval_dataset/sketch_retrieval/", self.image_path)
            ]
            
            _image_info = self.image_dict("")
            _image_info.update(
                {
                    "query_image_path": query_image_path,
                    "matched_image_path": matched_image_path,
                    "unmatched_image_path_lsit": unmatched_image_path_lsit,
                    "original_image_path": query_image_path
                }
            )

            self.images_info.append(_image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import random
        import copy
        import os
        num_choices = 3
        question = "Please retrieve the most similar scene to the query in the candidate."

        query_image_path = image_info["query_image_path"]
        match_image_path = image_info["matched_image_path"]

        candidate_image_list = image_info["unmatched_image_path_lsit"][:num_choices - 1]

        all_image_list = [query_image_path]

        gt_index = random.randrange(0, num_choices)
        candidate_image_list.insert(gt_index, match_image_path)
        all_image_list.extend(candidate_image_list)
        labels = ["Query Image", "Candidate 1", "Candidate 2", "Candidate 3"]

        BaseDataset.exist_or_mkdir(save_image_path)
        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_five_images(all_image_list, labels, merge_image_path, dpi=200)
            
        qa_json = {
            "question": question,
            "num_wrong_choices": num_choices - 1,
            # "wrong_choices_list": wrong_choices_list,
            "gt": labels[1:][gt_index],
            "choice_list": labels[1:],
            "gt_index": gt_index
        }

        qa_json["merge_image_path"] = merge_image_path

        qa_json["original_image_path"] = all_image_list

        qa_json["query_image_path"] = all_image_list[0]
        qa_json["choice_image_path"] = all_image_list[1:]
        qa_json["original_image_name"] = labels

        return qa_json


class DomainNet_retrieval(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/sketch_retrieval/meta_info.json",
        "image_path": "/path/to/sketch_retrieval/",
        "sampling_num": 100,
        "visual_input_component": ["natural_image", "sketch_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        anno_info = mmcv.load(self.anno_path)

        self.images_info = []
        for image_info in anno_info["images"]:
            if image_info["source"] != "DomainNet":
                continue
            
            query_image_path = image_info["image_query"].replace("/home/wenbo/Documents/data/LLM_eval_dataset/sketch_retrieval/", self.image_path)
            matched_image_path = image_info["image_candidate1"].replace("/home/wenbo/Documents/data/LLM_eval_dataset/sketch_retrieval/", self.image_path)
            unmatched_image_path_lsit = [
                image_info["image_candidate2"].replace("/home/wenbo/Documents/data/LLM_eval_dataset/sketch_retrieval/", self.image_path),
                image_info["image_candidate3"].replace("/home/wenbo/Documents/data/LLM_eval_dataset/sketch_retrieval/", self.image_path)
            ]
            
            _image_info = self.image_dict("")
            _image_info.update(
                {
                    "query_image_path": query_image_path,
                    "matched_image_path": matched_image_path,
                    "unmatched_image_path_lsit": unmatched_image_path_lsit,
                    "original_image_path": query_image_path
                }
            )

            self.images_info.append(_image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import random
        import copy
        import os
        num_choices = 3
        question = "Please retrieve the most similar scene to the query in the candidate."

        query_image_path = image_info["query_image_path"]
        match_image_path = image_info["matched_image_path"]

        candidate_image_list = image_info["unmatched_image_path_lsit"][:num_choices - 1]

        all_image_list = [query_image_path]

        gt_index = random.randrange(0, num_choices)
        candidate_image_list.insert(gt_index, match_image_path)
        all_image_list.extend(candidate_image_list)
        labels = ["Query Image", "Candidate 1", "Candidate 2", "Candidate 3"]

        BaseDataset.exist_or_mkdir(save_image_path)
        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_five_images(all_image_list, labels, merge_image_path, dpi=200)
            
        qa_json = {
            "question": question,
            "num_wrong_choices": num_choices - 1,
            # "wrong_choices_list": wrong_choices_list,
            "gt": labels[1:][gt_index],
            "choice_list": labels[1:],
            "gt_index": gt_index
        }

        qa_json["merge_image_path"] = merge_image_path

        qa_json["original_image_path"] = all_image_list
        qa_json["query_image_path"] = all_image_list[0]
        qa_json["choice_image_path"] = all_image_list[1:]

        qa_json["original_image_name"] = labels

        return qa_json
