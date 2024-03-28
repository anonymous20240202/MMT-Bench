import random
from pathlib import Path
from collections import defaultdict

from base_dataset import BaseDataset

from tqdm import tqdm


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
    imgs = [mpimg.imread(img_path) for img_path in image_paths]

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


class CUB220_2011_retrieval(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/samples/Text_to_image/metadata_info_new.json",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import mmcv
        import copy
        anno_info = mmcv.load(self.anno_path)

        images_info = []

        for image_info in anno_info["image"]:
            if image_info["source"] != "CUB220_2011":
                continue
            
            images_info.append(image_info)

        self.images_info = []
        for image_info in images_info:
            
            label = image_info["label"]

            _temp_images_info = copy.deepcopy(images_info)
            _temp_images_info.remove(image_info)

            matched_image_path = image_info["original_image_path"]
            candidate_image_path = [_image_info["original_image_path"] for _image_info in random.sample(_temp_images_info, 3)]

            new_image_info = self.image_dict("")

            new_image_info.update(
                {
                    "original_image_path": matched_image_path,
                    "gt_image_path": matched_image_path,
                    "candiate_image_path": candidate_image_path,
                    "label": label,
                }
            )

            self.images_info.append(new_image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import random
        import copy
        import os
        num_choices = 4
        question = f"Please find the most relevant picture among the candidate images for this description.\nDescription: {image_info['label']}"

        # query_image_path = image_info["gt_image_path"]
        match_image_path = image_info["gt_image_path"]

        candidate_image_list = image_info["candiate_image_path"][:num_choices - 1]

        all_image_list = []

        gt_index = random.randrange(0, num_choices)
        candidate_image_list.insert(gt_index, match_image_path)
        all_image_list.extend(candidate_image_list)
        labels = ["Candidate 1", "Candidate 2", "Candidate 3", "Candidate 4"]

        BaseDataset.exist_or_mkdir(save_image_path)
        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_five_images(all_image_list, labels, merge_image_path, dpi=200)
            
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


class Oxford_102_flower_retrieval(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/samples/Text_to_image/metadata_info_new.json",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import mmcv
        import copy
        import os
        anno_info = mmcv.load(self.anno_path)

        images_info = []

        for image_info in anno_info["image"]:
            if image_info["source"] != "Oxford_102_flower":
                continue
                
            if not os.path.exists(image_info["original_image_path"]):
                continue
            
            images_info.append(image_info)

        self.images_info = []
        for image_info in images_info:
            
            label = image_info["label"]

            _temp_images_info = copy.deepcopy(images_info)
            _temp_images_info.remove(image_info)

            matched_image_path = image_info["original_image_path"]
            candidate_image_path = [_image_info["original_image_path"] for _image_info in random.sample(_temp_images_info, 3)]

            new_image_info = self.image_dict("")

            new_image_info.update(
                {
                    "original_image_path": matched_image_path,
                    "gt_image_path": matched_image_path,
                    "candiate_image_path": candidate_image_path,
                    "label": label,
                }
            )

            self.images_info.append(new_image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import random
        import copy
        import os
        num_choices = 4
        question = f"Please find the most relevant picture among the candidate images for this description.\nDescription: {image_info['label']}"

        # query_image_path = image_info["gt_image_path"]
        match_image_path = image_info["gt_image_path"]

        candidate_image_list = image_info["candiate_image_path"][:num_choices - 1]

        all_image_list = []

        gt_index = random.randrange(0, num_choices)
        candidate_image_list.insert(gt_index, match_image_path)
        all_image_list.extend(candidate_image_list)
        labels = ["Candidate 1", "Candidate 2", "Candidate 3", "Candidate 4"]

        BaseDataset.exist_or_mkdir(save_image_path)
        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_five_images(all_image_list, labels, merge_image_path, dpi=200)
            
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