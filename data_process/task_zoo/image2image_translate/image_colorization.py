import random
from pathlib import Path
from collections import defaultdict

from base_dataset import BaseDataset

from tqdm import tqdm
import mmcv

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np

from PIL import Image
import os


def plot_and_save_five_images(image_paths, labels, save_path, dpi):
    """
    Plots five images side by side with their respective labels and saves the plot to a specified path.

    Parameters:
    image_paths (list of str): List of file paths for the images.
    labels (list of str): List of labels for the images.
    save_path (str): Path to save the combined image plot.
    dpi (int): Resolution of the saved image.
    """

    if len(image_paths) != 4 or len(labels) != 4:
        raise ValueError("Five image paths and labels are required.")

    # 读取图片
    imgs = []
    for image_path in image_paths:
        _img = mpimg.imread(image_path)

        if len(_img.shape) == 3 and _img.shape[2] == 3:
            pass
        else:
            _img = np.stack((_img,)*3, axis=-1)
        
        imgs.append(_img)
    

    # imgs = [mpimg.imread(img_path)  for img_path in image_paths]

    # 创建图形和子图
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    # 在子图上显示图片和标签
    for ax, img, label in zip(axes, imgs, labels):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')  # 关闭坐标轴

    # 调整子图之间的间隔
    plt.subplots_adjust(wspace=0.5)

    # 保存图形到指定路径
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)


def convert_image_style(input_image_path, style, output_image_path):
    """
    Convert the style of an image and save it to a new path.

    Parameters:
    - input_image_path: str. The file path of the input image.
    - style: str. The desired style to apply ('cold', 'warm', 'sepia').
    - output_image_path: str. The file path where the converted image will be saved.
    """
    # Load the image
    image = Image.open(input_image_path)
    
    # Remove alpha channel if present
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Convert to numpy array for manipulation
    image_array = np.array(image)
    
    # Apply the selected style
    if style == 'cold':
        image_array[:, :, 0] = (image_array[:, :, 0] * 0.9).clip(0, 255)  # Red channel
        image_array[:, :, 2] = (image_array[:, :, 2] * 1.1).clip(0, 255)  # Blue channel
    elif style == 'warm':
        image_array[:, :, 0] = (image_array[:, :, 0] * 1.1).clip(0, 255)  # Red channel
        image_array[:, :, 2] = (image_array[:, :, 2] * 0.9).clip(0, 255)  # Blue channel
    elif style == 'sepia':
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                 [0.349, 0.686, 0.168],
                                 [0.272, 0.534, 0.131]])
        sepia_img = image_array.dot(sepia_filter.T)
        sepia_img = np.clip(sepia_img, 0, 255)  # Ensure values are within [0, 255]
        image_array = sepia_img.astype(np.uint8)
    else:
        raise ValueError("Style not recognized. Choose 'cold', 'warm', or 'sepia'.")
    
    # Convert back to an image
    processed_image = Image.fromarray(image_array)
    
    # Save the modified image
    processed_image.save(output_image_path)
    
    return output_image_path


class Image_Colorization_Dataset(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/samples/Image_colorization/Image Colorization Dataset/metadata_info_new.json",
        "save_image_path": "/path/to/lvlm_evaluation/data_process/taskonomy_evaluation_data/image-to-image_translation/image_colorization/images",
        "sampling_num": 100,
        "visual_input_component": ["image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = []
        anno_info = mmcv.load(self.anno_path)

        for image_info in tqdm(anno_info["image"]):
            image_info["source"] = self.dataset_name

            warm_image_path = os.path.join(self.save_image_path, self.new_image_name())
            image_info["warm_image_path"] = convert_image_style(image_info["original_image_path"], "warm", warm_image_path)

            cold_image_path = os.path.join(self.save_image_path, self.new_image_name())
            image_info["cold_image_path"] = convert_image_style(image_info["original_image_path"], "cold", cold_image_path)

            sepia_image_path = os.path.join(self.save_image_path, self.new_image_name())
            image_info["sepia_image_path"] = convert_image_style(image_info["original_image_path"], "sepia", sepia_image_path)

            self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import random
        import copy
        import os
        num_choices = 4
        question = "The following images are from the same picture, which consists of four styles: grayscale, original, warm, and sepia. Which one is the original picture?"

        all_image_list = [image_info["grayscale_image_path"], image_info["warm_image_path"], image_info["sepia_image_path"]]
        random.shuffle(all_image_list)

        gt_index = random.randrange(0, num_choices)
        all_image_list.insert(gt_index, image_info["original_image_path"])
        labels = ["Candidate 1", "Candidate 2", "Candidate 3", "Candidate 4"]

        BaseDataset.exist_or_mkdir(save_image_path)
        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_five_images(all_image_list, labels, merge_image_path, dpi=300)
            
        qa_json = {
            "question": question,
            "num_wrong_choices": 3,
            "gt": labels[gt_index],
            "choice_list": labels,
            "gt_index": gt_index
        }

        qa_json["merge_image_path"] = merge_image_path

        qa_json["original_image_path"] = all_image_list
        qa_json["choice_image_path"] = all_image_list
        qa_json["original_image_name"] = labels

        return qa_json


class Landscape_color_and_grayscale_images(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/samples/Image_colorization/Landscape color and grayscale images/metadata_info_new.json",
        "save_image_path": "/path/to/lvlm_evaluation/data_process/taskonomy_evaluation_data/image-to-image_translation/image_colorization/images",
        "sampling_num": 100,
        "visual_input_component": ["image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = []
        anno_info = mmcv.load(self.anno_path)

        for image_info in tqdm(anno_info["image"]):
            image_info["source"] = self.dataset_name

            warm_image_path = os.path.join(self.save_image_path, self.new_image_name())
            image_info["warm_image_path"] = convert_image_style(image_info["original_image_path"], "warm", warm_image_path)

            cold_image_path = os.path.join(self.save_image_path, self.new_image_name())
            image_info["cold_image_path"] = convert_image_style(image_info["original_image_path"], "cold", cold_image_path)

            sepia_image_path = os.path.join(self.save_image_path, self.new_image_name())
            image_info["sepia_image_path"] = convert_image_style(image_info["original_image_path"], "sepia", sepia_image_path)

            self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import random
        import copy
        import os
        num_choices = 4
        question = "The following images are from the same picture, which consists of four styles: grayscale, original, warm, and sepia. Which one is the original picture?"

        all_image_list = [image_info["grayscale_image_path"], image_info["warm_image_path"], image_info["sepia_image_path"]]
        random.shuffle(all_image_list)

        gt_index = random.randrange(0, num_choices)
        all_image_list.insert(gt_index, image_info["original_image_path"])
        labels = ["Candidate 1", "Candidate 2", "Candidate 3", "Candidate 4"]

        BaseDataset.exist_or_mkdir(save_image_path)
        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_five_images(all_image_list, labels, merge_image_path, dpi=300)
            
        qa_json = {
            "question": question,
            "num_wrong_choices": 3,
            # "wrong_choices_list": wrong_choices_list,
            "gt": labels[gt_index],
            "choice_list": labels,
            "gt_index": gt_index
        }

        qa_json["merge_image_path"] = merge_image_path

        qa_json["original_image_path"] = all_image_list
        qa_json["choice_image_path"] = all_image_list
        qa_json["original_image_name"] = labels

        return qa_json
