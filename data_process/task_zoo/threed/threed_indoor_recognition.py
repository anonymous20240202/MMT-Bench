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


def plot_and_save_images(image_paths, labels, save_path, dpi):
    # 读取图片
    imgs = [mpimg.imread(img_path) for img_path in image_paths]

    # 创建图形和子图
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

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


class ScanObjectNN(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to1/evaluation_data/finish/threeD_perception/threeD_indoor_classification/metadata_info.json",
        "sampling_num": 200,
        "visual_input_component": ["indoor_CAD_Point"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        
        anno_info = mmcv.load(self.anno_path)
        self.category_space = list()

        for image_info in anno_info["images"]:

            original_image_path = [
                image_info["image_file_view1"],
                image_info["image_file_view2"],
                image_info["image_file_view3"],
                image_info["image_file_view4"],
                image_info["image_file_view5"],
                image_info["image_file_view6"],
            ]

            wrong = False
            for image_path in original_image_path:
                if not os.path.exists(image_path):
                    wrong = True
            
            if wrong:
                continue
                    
            image_info["source"] = self.dataset_name
            image_info["original_image_path"] = original_image_path
            image_info["category"] = " ".join(image_info["point_label_name"].split("_"))

            del image_info["point_feature"]

            self.category_space.append(image_info["category"])

            self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = list(set(self.category_space))

    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the category of the point cloud based on the multi-view of the point cloud?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "vase",
                "question": question,
                "wrong_choices_list": ["person", "toilet", "table"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["category"],
                "question": question,
            }
        }

        label_list = ["View 1", "View 2", "View 3", "View 4", "View 5", "View 6"]
        image_path_list = image_info["original_image_path"]

        BaseDataset.exist_or_mkdir(save_image_path)
        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_images(image_path_list, label_list, merge_image_path, dpi=200)

        user_prompt = json.dumps(input_json)

        i = 0
        while i <= 10:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json, question=question)
                qa_json["merge_image_path"] = merge_image_path
                break
            except:
                i += 1

        return qa_json
