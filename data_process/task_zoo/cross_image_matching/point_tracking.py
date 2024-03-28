from pathlib import Path
from collections import OrderedDict

from base_dataset import BaseDataset
import mmcv

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from prompt.utils import *


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
    plt.subplots_adjust(wspace=0.1)

    # 保存图形到指定路径
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

class tapvid_davis(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/lvlm_evaluation/data_process/data/point_tracking/tap_vid/tapvid_davis/tapvid_davis.pkl",
        "save_image_path": "/path/to/lvlm_evaluation/data_process/taskonomy_evaluation_data/cross_image_matching/point_tracking/images",
        "sampling_num": 100,
        "url": "internet",
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx",
        "sampling_range": 50,
        "sampling_ratio": 0.01,

    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        anno_info = mmcv.load(self.anno_path)

        for _, video_info in tqdm(anno_info.items()):
            num_frames = video_info["video"].shape[0]
            num_points = video_info["points"].shape[0]

            sampling_pair_list = random_frame_sampling(num_frames=num_frames, sampling_range=self.sampling_range, sampling_ratio=self.sampling_ratio)

            for sampling_pair in sampling_pair_list:
                point_id = random.randint(0, num_points-1)
                point_1 = video_info["points"][point_id][sampling_pair[0]]
                point_2 = video_info["points"][point_id][sampling_pair[1]]
            
                image_1 = video_info["video"][sampling_pair[0]]
                image_2 = video_info["video"][sampling_pair[1]]

                out_image_name_1 = os.path.join(self.save_image_path, self.new_image_name())
                out_image_name_2 = os.path.join(self.save_image_path, self.new_image_name())
                self.save_rgb_image(image_1[:, :, ::-1], out_image_name_1)
                self.save_rgb_image(image_2[:, :, ::-1], out_image_name_2)

                image_info = self.image_dict(out_image_name_1)

                image_info.update(
                    {
                        "original_image_path": [out_image_name_1, out_image_name_2],
                        "point1": point_1.tolist(),
                        "point2": point_2.tolist()
                    }
                )
            
                self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import json
        import mmcv

        num_choices = 4
        width, height = image_info["width"], image_info["height"]

        point1 = [round(image_info["point1"][0], 3), round(image_info["point1"][1], 3)]
        point2 = [round(image_info["point2"][0], 3), round(image_info["point2"][1], 3)]

        question = f"What is the position coordinates of the point with coordinates ({point1}) in Frame 1 within the Frame 2? Note that the width of the input RGB image is {width} and the height is {height}."

        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_two_images(image_info["original_image_path"][0], image_info["original_image_path"][1], "Frame 1", "Frame 2", merge_image_path, dpi=300)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = []
                for i in range(num_choices - 1):
                    wrong_choices_list.append([round(random.uniform(0, 1), 3), round(random.uniform(0, 1), 3)])
                    
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": point2,
                    "question": question,
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                qa_json["merge_image_path"] = merge_image_path
                qa_json["original_image_name"] = ["Frame 1", "Frame 2"]
                break
            except:
                i += 1

        return qa_json


class tapvid_rgb_stacking(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/lvlm_evaluation/data_process/data/point_tracking/tap_vid/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl",
        "save_image_path": "/path/to/lvlm_evaluation/data_process/taskonomy_evaluation_data/cross_image_matching/point_tracking/images",
        "sampling_num": 100,
        "url": "internet",
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx",
        "sampling_range": 50,
        "sampling_ratio": 0.0001,
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        anno_info = mmcv.load(self.anno_path)

        for video_info in tqdm(anno_info):
            num_frames = video_info["video"].shape[0]
            num_points = video_info["points"].shape[0]

            sampling_pair_list = random_frame_sampling(num_frames=num_frames, sampling_range=self.sampling_range, sampling_ratio=self.sampling_ratio)

            for sampling_pair in sampling_pair_list:
                point_id = random.randint(0, num_points-1)
                point_1 = video_info["points"][point_id][sampling_pair[0]]
                point_2 = video_info["points"][point_id][sampling_pair[1]]
            
                image_1 = video_info["video"][sampling_pair[0]]
                image_2 = video_info["video"][sampling_pair[1]]

                out_image_name_1 = os.path.join(self.save_image_path, self.new_image_name())
                out_image_name_2 = os.path.join(self.save_image_path, self.new_image_name())
                self.save_rgb_image(image_1, out_image_name_1)
                self.save_rgb_image(image_2, out_image_name_2)

                image_info = self.image_dict(out_image_name_1)

                image_info.update(
                    {
                        "original_image_path": [out_image_name_1, out_image_name_2],
                        "point1": point_1.tolist(),
                        "point2": point_2.tolist()
                    }
                )
            
                self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import json
        import mmcv

        num_choices = 4
        width, height = image_info["width"], image_info["height"]

        point1 = [round(image_info["point1"][0], 3), round(image_info["point1"][1], 3)]
        point2 = [round(image_info["point2"][0], 3), round(image_info["point2"][1], 3)]

        question = f"What is the position coordinates of the point with coordinates ({point1}) in Frame 1 within the Frame 2? Note that the width of the input RGB image is {width} and the height is {height}."

        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_two_images(image_info["original_image_path"][0], image_info["original_image_path"][1], "Frame 1", "Frame 2", merge_image_path, dpi=300)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = []
                for i in range(num_choices - 1):
                    wrong_choices_list.append([round(random.uniform(0, 1), 3), round(random.uniform(0, 1), 3)])
                    
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": point2,
                    "question": question,
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                qa_json["merge_image_path"] = merge_image_path

                qa_json["original_image_name"] = ["Frame 1", "Frame 2"]
                break
            except:
                i += 1

        return qa_json