from pathlib import Path
from collections import OrderedDict

from base_dataset import BaseDataset

import random
import mmcv
import os

from tqdm import tqdm
import mmcv
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import random
from pathlib import Path
from collections import defaultdict

from base_dataset import BaseDataset

from tqdm import tqdm
import mmcv
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from prompt.utils import *


def plot_and_save_images(image_paths, labels, save_path, dpi):
    # 读取图片
    imgs = [mpimg.imread(img_path) for img_path in image_paths]

    # 创建图形和子图
    fig, axes = plt.subplots(3, 3, figsize=(16, 9))

    # 在子图上显示图片和标签
    for i, ax in enumerate(axes.flat):
        if i < len(imgs):  # 避免在不足 2x3 个子图时出错
            ax.imshow(imgs[i])
            ax.set_title(labels[i])
            ax.axis('off')  # 关闭坐标轴

    # 调整子图之间的间隔
    plt.subplots_adjust(wspace=0.1)

    # 保存图形到指定路径
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)


class MeViS(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/TaskOnomy_Evaluation/temporal_understanding/MeViS/metadata_info.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "Visual Genome Relation tests the understanding of objects' relation in complex natural scenes. Given an image and a constituent relation of the form X relation Y, we test whether the model can pick the correct order."
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        
        anno_info = mmcv.load(self.anno_path)

        for image_info in anno_info["images"]:
            caption = image_info["caption"]

            frames_list = image_info["frames"]

            if len(frames_list) < 9:
                continue
            n = 9
            original_image_path = [frames_list[0]]
            original_image_path.extend(sorted(random.sample(frames_list[1:-1], n - 2)))
            original_image_path.append(frames_list[-1])
            
            bbox = image_info["boxes"][frames_list.index(original_image_path[4])]

            original_image_path = [_path[1:] for _path in original_image_path]

            image_info["bbox"] = bbox
            image_info["original_image_path"] = original_image_path

            image_info["source"] = self.dataset_name

            self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        num_choices = 4

        original_image_path = image_info["original_image_path"]

        width, height = image_info["width"], image_info["height"]

        image_labels = ["Frame 1", "Frame 2", "Frame 3", "Frame 4","Frame 5" ,"Frame 6", "Frame 7", "Frame 8", "Frame 9"]

        BaseDataset.exist_or_mkdir(save_image_path)
        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())
        plot_and_save_images(original_image_path, image_labels, merge_image_path, dpi=200)

        caption = image_info["caption"]
        question = f"I have provided several frames from a video, and I will also provide a caption. Provide the output for the detected area in the format [x, y, w, h]. This format represents the bounding box, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as its width and height. Note that the width of the input image is {width} and the height is {height}.\nCAPTION: {caption}"

        bbox = image_info["bbox"]
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
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
                break
            except:
                i += 1
        
        qa_json["merge_image_path"] = merge_image_path

        return qa_json
