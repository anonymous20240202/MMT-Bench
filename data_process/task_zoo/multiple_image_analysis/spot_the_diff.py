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


class spot_the_diff(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/emotional_quotient_test/facial-emotion-recognition-dataset/data",
        "anno_path": "/path/to/taskonomy_data/multiimage_analysis/spot_difference/spot-the-diff/metadata_info.json",
        "sampling_num": 200,
        "visual_input_component": [
            "natural_image"
        ],
        "dataset_description": "Spot-the-diff is a dataset consisting of 13,192 image pairs along with corresponding human provided text annotations stating the differences between the two images.",
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)

        for image_info in anno_info["images"]:
            image_info["original_image_path"] = [image_path.replace("lustre", "petrelfs") for image_path in image_info["image_path"]]
            image_info["source"] = self.dataset_name
            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import os
        import random
        import json
        width, height = image_info["width"], image_info["height"]
        num_choices = 4
        question = f"The following is a description of the differences between two pictures. Which one is incorrect?"
        image_1_path = image_info["original_image_path"][0]
        image_2_path = image_info["original_image_path"][1]

        BaseDataset.exist_or_mkdir(save_image_path)
        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_two_images(image_1_path, image_2_path, "Image 1", "Image 2", merge_image_path, dpi=70)

        num_choices = len(image_info["image_label"]) + 1
        right_choices_list = len(image_info["image_label"])
        chatgpt_prompt = "I will provide you with several correct descriptions of two images, and I want you to generate a new description that is unrelated to the correct descriptions. Please return it in JSON format with the key 'wrong'."

        i = 0
        while i <= 10:
            try:
                wrong_str = BaseDataset.openai_generate(chatgpt_prompt, json.dumps(image_info["image_label"]))["wrong"]
                assert isinstance(wrong_str, str)
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": wrong_str,
                    "question": question,
                    "wrong_choices_list": image_info["image_label"]
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)

                qa_json["merge_image_path"] = merge_image_path
                break
            except:
                i += 1

        return qa_json
