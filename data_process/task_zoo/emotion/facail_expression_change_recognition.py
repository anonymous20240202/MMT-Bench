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


class emotion_change(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/emotional_quotient_test/facial-emotion-recognition-dataset/data",
        "sampling_num": 100,
        "visual_input_component": [
            "natural_image"
        ],
        "dataset_description": "xxx",
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        category_space = []

        for person_path in Path(self.image_path).iterdir():
            image_path_list = []
            sample_list = []
            for image_path in person_path.iterdir():
                image_path_list.append(image_path)

                category_space.append(image_path.stem)

            for i in range(100):
                l = random.sample(image_path_list, 2)
                if l not in sample_list:
                    sample_list.append(l)
                
            for image_pair in sample_list:
                image_info = self.image_dict("")
                image_info["original_image_path"] = [str(image_pair[0]), str(image_pair[1])]
                image_info["category"] = [image_pair[0].stem, image_pair[1].stem]

                self.images_info.append(image_info)

        self.dataset_info["category_space"] = list(set(category_space))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import os
        import random
        width, height = image_info["width"], image_info["height"]
        num_choices = 4
        question = f"What is the change of expression from Image 1 to Image 2?"
        image_1_path = image_info["original_image_path"][0]
        image_2_path = image_info["original_image_path"][1]

        BaseDataset.exist_or_mkdir(save_image_path)
        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_two_images(image_1_path, image_2_path, "Image 1", "Image 2", merge_image_path, dpi=200)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = []
                for i in range(num_choices - 1):
                    pair_info = random.sample(dataset_info["category_space"], 2)

                    wrong_choices_list.append(f"{pair_info[0]} to {pair_info[1]}")

                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": f"{image_info['category'][0]} to {image_info['category'][1]}",
                    "question": question,
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                qa_json["merge_image_path"] = merge_image_path
                break
            except:
                i += 1

        return qa_json


# class CKPlus48(BaseDataset):
#     DATA_METAINFO = {
#         "anno_path": "/path/to/TaskOnomy_Evaluation/emotional_quotient_test/facial_expression_change_recognition/metadata_info.json",
#         "sampling_num": 30,
#         "visual_input_component": [
#             "natural_image",
#             "grayscale",
#             "low-resolution"
#         ],
#         "dataset_description": "This dataset comes from kaggle: https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset",
#     }
    
#     def parse_dataset_info(self):
#         super().parse_dataset_info()
    
#     def parse_images_info(self):
#         self.images_info = list()

#         anno_info = mmcv.load(self.anno_path)
#         for image_info in anno_info["images"]:

#             if image_info["source"] != "CKPlus48":
#                 continue

#             image_info["source"] = self.dataset_name

#             image_info["original_image_path"] = [_[1:] for _ in image_info["original_image_path"]]
#             self.images_info.append(image_info)
        
#         self.dataset_info["category_space"] = anno_info["dataset_list"][0]["category_space"]
    
#     @staticmethod
#     def generate_qa(image_info, dataset_info, save_image_path):
#         import os
#         import random
#         width, height = image_info["width"], image_info["height"]
#         num_choices = 4
#         question = f"What is the change of expression from Image 1 to Image 2?"
#         image_1_path = image_info["original_image_path"][0]
#         image_2_path = image_info["original_image_path"][1]

#         BaseDataset.exist_or_mkdir(save_image_path)
#         merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

#         plot_and_save_two_images(image_1_path, image_2_path, "Image 1", "Image 2", merge_image_path, dpi=200)

#         i = 0
#         while i <= 10:
#             try:
#                 wrong_choices_list = []
#                 for i in range(num_choices):
#                     wrong_choices_list.append(random.sample())
                    
#                 qa_json = {
#                     "num_wrong_choices": num_choices - 1,
#                     "gt": bbox,
#                     "question": question,
#                     "wrong_choices_list": wrong_choices_list
#                 }
#                 qa_json = BaseDataset.post_process(qa_json, question=question)
#                 qa_json["merge_image_path"] = merge_image_path
#                 break
#             except:
#                 i += 1

#         return qa_json


class ferg_db(BaseDataset):

    DATA_METAINFO = {
        "anno_path": "/path/to/TaskOnomy_Evaluation/emotional_quotient_test/facial_expression_change_recognition/metadata_info.json",
        "sampling_num": 100,
        "visual_input_component": [
            "synthetic_image"
        ],
        "dataset_description": "This dataset comes from kaggle: https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)
        for image_info in anno_info["images"]:

            if image_info["source"] != "ferg_db":
                continue
            
            image_info["original_image_path"] = [_[1:] for _ in image_info["original_image_path"]]
            image_info["source"] = self.dataset_name
            self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = anno_info["dataset_list"][0]["category_space"]
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import os
        import random
        width, height = image_info["width"], image_info["height"]
        num_choices = 4
        question = f"What is the change of expression from Image 1 to Image 2?"
        image_1_path = image_info["original_image_path"][0]
        image_2_path = image_info["original_image_path"][1]

        BaseDataset.exist_or_mkdir(save_image_path)
        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_two_images(image_1_path, image_2_path, "Image 1", "Image 2", merge_image_path, dpi=200)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = []
                for i in range(num_choices - 1):
                    pair_info = random.sample(dataset_info["category_space"], 2)

                    wrong_choices_list.append(f"{pair_info[0]} to {pair_info[1]}")

                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": f"{image_info['category'][0]} to {image_info['category'][1]}",
                    "question": question,
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                qa_json["merge_image_path"] = merge_image_path
                break
            except:
                i += 1

        return qa_json
    