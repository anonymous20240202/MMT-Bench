from pathlib import Path

import struct
import random
import numpy as np

import mmcv

from base_dataset import BaseDataset

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_and_save_two_images(image_path1, image_path2, label1, label2, save_path, dpi):
    """
    Plots two images side by side with their respective labels and saves the plot to a specified path.

    Parameters:
    image_path1 (str): File path of the first image.
    image_path2 (str): File path of the second image.
    label1 (str): Label for the first image.
    label2 (str): Label for the second image.
    save_path (str): Path to save the combined image plot.
    """

    # 读取两张图片
    img1 = mpimg.imread(image_path1)
    img2 = mpimg.imread(image_path2)

    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # 在子图上显示图片和标签
    ax1.imshow(img1)
    ax1.set_title(label1)
    ax1.axis('off')  # 关闭坐标轴

    ax2.imshow(img2)
    ax2.set_title(label2)
    ax2.axis('off')  # 关闭坐标轴

    # 调整子图之间的间隔
    plt.subplots_adjust(wspace=0.3)

    # 保存图形到指定路径
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)


class image_alike(BaseDataset):
    DATA_METAINFO = {
        "image_path": "",
        "anno_path": "/path/to/image_smilarity/meta_info.json",
        "sampling_num": 100,
        "visual_input_component": [
            "natural_image"
        ],
        "dataset_description": "'images' folder contains pairs of images that look alike (each image is separated from its look alike). The 'validate.csv' contains the look alike images uid from 'image_a' and 'image_b'.",
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)

        for image_info in anno_info["images"]:
            image_info["original_image_path"] = [
                image_info["image1"].replace("/home/wenbo/Documents/data/LLM_eval_dataset/image_smilarity", "/path/to/image_smilarity"),
                image_info["image2"].replace("/home/wenbo/Documents/data/LLM_eval_dataset/image_smilarity", "/path/to/image_smilarity")
                ]
            image_info["source"] = self.dataset_name
            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import os
        import random
        import json
        width, height = image_info["width"], image_info["height"]
        num_choices = 2
        question = f"Are there any similarities between the two pictures?"
        image_1_path = image_info["original_image_path"][0]
        image_2_path = image_info["original_image_path"][1]

        BaseDataset.exist_or_mkdir(save_image_path)
        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_two_images(image_1_path, image_2_path, "Image 1", "Image 2", merge_image_path, dpi=70)

        if image_info["judgment"] == 1:
            gt = "Yes"
            wrong_choices_list = ["No"]
        else:
            gt = "No"
            wrong_choices_list = ["Yes"]
        i = 0
        while i <= 10:
            try:
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt,
                    "question": question,
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)

                qa_json["merge_image_path"] = merge_image_path
                break
            except:
                i += 1

        return qa_json


class Totally_Looks_Like_Data(BaseDataset):
    DATA_METAINFO = {
        "image_path": "",
        "anno_path": "/path/to/image_smilarity/meta_info.json",
        "sampling_num": 100,
        "visual_input_component": [
            "natural_image"
        ],
        "dataset_description": "'images' folder contains pairs of images that look alike (each image is separated from its look alike). The 'validate.csv' contains the look alike images uid from 'image_a' and 'image_b'.",
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)

        for image_info in anno_info["images"]:
            image_info["original_image_path"] = [
                image_info["image1"].replace("/home/wenbo/Documents/data/LLM_eval_dataset/image_smilarity", "/path/to/image_smilarity"),
                image_info["image2"].replace("/home/wenbo/Documents/data/LLM_eval_dataset/image_smilarity", "/path/to/image_smilarity")
            ]
            image_info["source"] = self.dataset_name
            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import os
        import random
        import json
        width, height = image_info["width"], image_info["height"]
        num_choices = 2
        question = f"Are there any similarities between the two pictures?"
        image_1_path = image_info["original_image_path"][0]
        image_2_path = image_info["original_image_path"][1]

        BaseDataset.exist_or_mkdir(save_image_path)
        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_two_images(image_1_path, image_2_path, "Image 1", "Image 2", merge_image_path, dpi=70)

        if image_info["judgment"] == 1:
            gt = "Yes"
            wrong_choices_list = ["No"]
        else:
            gt = "No"
            wrong_choices_list = ["Yes"]
        i = 0
        while i <= 10:
            try:
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt,
                    "question": question,
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)

                qa_json["merge_image_path"] = merge_image_path
                break
            except:
                i += 1

        return qa_json
