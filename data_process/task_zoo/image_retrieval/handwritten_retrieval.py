import random
from pathlib import Path
from collections import defaultdict

from base_dataset import BaseDataset

from tqdm import tqdm

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

    if len(image_paths) != 5 or len(labels) != 5:
        raise ValueError("Five image paths and labels are required.")

    imgs = []
    for image_path in image_paths:
        _img = mpimg.imread(image_path)

        if len(_img.shape) == 3 and _img.shape[2] == 3:
            pass
        else:
            _img = np.stack((_img,)*3, axis=-1)
        
        imgs.append(_img)

    # 创建图形和子图
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    # 在子图上显示图片和标签
    for ax, img, label in zip(axes, imgs, labels):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')  # 关闭坐标轴

    # 调整子图之间的间隔
    plt.subplots_adjust(wspace=0.2)

    # 保存图形到指定路径
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)


class iam_handwritten_retrieval(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/hand_written/iam/data",
        "sampling_num": 200,
        "visual_input_component": ["text-rich_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        query_id2image_path = defaultdict(list)
        for a_path in Path(self.image_path).iterdir():
            for image_path in a_path.iterdir():
                query_id2image_path[a_path.stem].append(str(image_path))
        
        self.images_info = list()
        for person_id in tqdm(query_id2image_path.keys()):
            # sample query
            query_image_path = random.choice(query_id2image_path[person_id])

            _temp_gallery = list(query_id2image_path[person_id]).copy()
            _temp_gallery.remove(query_image_path)
            if len(_temp_gallery) == 0:
                continue
            match_image_path = str(random.choice(_temp_gallery))

            unmatched_num = 5

            _temp_gallery = list(query_id2image_path.keys()).copy()
            _temp_gallery.remove(person_id)

            unmatched_person_id_list = random.sample(_temp_gallery, unmatched_num)
            unmatched_image_path = [str(random.choice(query_id2image_path[i])) for i in unmatched_person_id_list]

            image_info = self.image_dict("")
            image_info.update(
                {
                    "query_image_path": query_image_path,
                    "match_image_path": match_image_path,
                    "unmatched_image_path_list": unmatched_image_path,
                    "original_image_path": query_image_path
                }
            )

            self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import random
        import copy
        import os
        num_choices = 4
        question = "Please retrieve the most similar scene to the query in the candidate."

        query_image_path = image_info["query_image_path"]
        match_image_path = image_info["match_image_path"]

        candidate_image_list = image_info["unmatched_image_path_list"][:num_choices - 1]

        all_image_list = [query_image_path]

        gt_index = random.randrange(0, num_choices)
        candidate_image_list.insert(gt_index, match_image_path)
        all_image_list.extend(candidate_image_list)
        labels = ["Query Image", "Candidate 1", "Candidate 2", "Candidate 3", "Candidate 4"]

        BaseDataset.exist_or_mkdir(save_image_path)
        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_five_images(all_image_list, labels, merge_image_path, dpi=300)
            
        qa_json = {
            "question": question,
            "num_wrong_choices": 3,
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
