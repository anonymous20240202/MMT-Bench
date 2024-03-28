import numpy as np
import os
from base_dataset import BaseDataset
from PIL import Image
from collections import defaultdict
import cv2
from scipy import ndimage
from tqdm import tqdm

from prompt.utils import *

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


from PIL import Image
import matplotlib.pyplot as plt

def merge_images_with_titles(image1, title1, image2, title2, save_path):
    """
    Merge two images side by side, add titles to each, and save the merged image to a specified path.

    Parameters:
    image1 (str): Path to the first image.
    title1 (str): Title for the first image.
    image2 (str): Path to the second image.
    title2 (str): Title for the second image.
    save_path (str): Path where the merged image will be saved.

    Returns:
    None: The function saves the merged image to the specified path.
    """
    # Open the images
    img1 = Image.open(image1)
    img2 = Image.open(image2)

    # Resize images to the same height
    img1_height = img1.size[1]
    img2_height = img2.size[1]
    if img1_height != img2_height:
        img2 = img2.resize((int(img2.size[0] * img1_height / img2_height), img1_height), Image.ANTIALIAS)

    # Merge images
    total_width = img1.size[0] + img2.size[0]
    merged_image = Image.new('RGB', (total_width, img1_height))
    merged_image.paste(img1, (0, 0))
    merged_image.paste(img2, (img1.size[0], 0))

    # Save the merged image
    merged_image.save(save_path)

    return


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

    # # 显示图形
    # plt.show()


class DES(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/localization/salient_object_detection_rgbd/DES/RGB",
        "anno_path": "/path/to/TaskOnomy_Evaluation/localization/salient_object_detection_rgbd/DES/GT",
        "depth_path": "/path/to/TaskOnomy_Evaluation/localization/salient_object_detection_rgbd/DES/depth",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "Note that this dataset have no category space and no train set (only used for testing). We use foreground as the category name"
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
            original_depth_path = os.path.join(self.DATA_METAINFO['depth_path'], im.replace('.jpg', '.png'))
            img = cv2.imread(original_image_path)
            h, w = img.shape[:2]
            mask_path = os.path.join(self.DATA_METAINFO['anno_path'], im.replace('.jpg', '.png'))
            gt_mask = np.asarray(Image.open(mask_path))
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
            'original_depth_path': original_depth_path.replace('/cpfs01/user/linyuqi/datasets', ' /path/to'),
            'height': h,
            'width': w
            }
            self.images_info.append(info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        width, height = image_info["width"], image_info["height"]
        num_choices = 4
        question = f"The left image is RGB image and the right image is the corresponding depth map. Please detect the salient foreground object in this RGB image and represent them using a single bounding box. Provide the output for the detected area in the format [x, y, w, h]. This format represents the bounding box, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as its width and height. Note that the width of the input RGB image is {width} and the height is {height}."
        image_path = image_info["original_image_path"]
        depth_map_path = image_info["original_depth_path"]

        BaseDataset.exist_or_mkdir(save_image_path)
        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_two_images(image_path, depth_map_path, "RGB image", "Depth map", merge_image_path, dpi=200)

        bbox = image_info["boxes"]['foreground'][0]
        bbox = xyxy2xywh(bbox)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_bounding_box_from_single_bbox(bbox, width, height, num_choices-1)
                    
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": bbox,
                    "question": question,
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                qa_json["merge_image_path"] = merge_image_path
                break
            except:
                i += 1

        return qa_json
            
            
class NJU2K(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/TaskOnomy_Evaluation/localization/salient_object_detection_rgbd/NJU2K/RGB_left",
        "anno_path": "/path/to/TaskOnomy_Evaluation/localization/salient_object_detection_rgbd/NJU2K/GT",
        "depth_path": "/path/to/TaskOnomy_Evaluation/localization/salient_object_detection_rgbd/NJU2K/depth",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "Note that this dataset have no category space and no train set (only used for testing). We use foreground as the category name"
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
            original_depth_path = os.path.join(self.DATA_METAINFO['depth_path'], im.replace('.jpg', '.png'))
            img = cv2.imread(original_image_path)
            h, w = img.shape[:2]
            mask_path = os.path.join(self.DATA_METAINFO['anno_path'], im.replace('.jpg', '.png'))
            gt_mask = np.asarray(Image.open(mask_path))
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
            'original_depth_path': original_depth_path.replace('/cpfs01/user/linyuqi/datasets', ' /path/to'),
            'height': h,
            'width': w
            }
            self.images_info.append(info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        width, height = image_info["width"], image_info["height"]
        num_choices = 4
        question = f"The left image is RGB image and the right image is the corresponding depth map. Please detect the salient foreground object in this RGB image and represent them using a single bounding box. Provide the output for the detected area in the format [x, y, w, h]. This format represents the bounding box, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as its width and height. Note that the width of the input RGB image is {width} and the height is {height}."
        image_path = image_info["original_image_path"]
        depth_map_path = image_info["original_depth_path"]

        BaseDataset.exist_or_mkdir(save_image_path)
        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_two_images(image_path, depth_map_path, "RGB Image", "Depth Map", merge_image_path, dpi=200)

        bbox = image_info["boxes"]['foreground'][0]
        bbox = xyxy2xywh(bbox)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_bounding_box_from_single_bbox(bbox, width, height, num_choices-1)
                    
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": bbox,
                    "question": question,
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                qa_json["merge_image_path"] = merge_image_path
                qa_json["original_image_path"] = [image_path, depth_map_path]
                qa_json["original_image_name"] = ["RGB Image", "Depth Map"]
                break
            except:
                i += 1

        return qa_json
            