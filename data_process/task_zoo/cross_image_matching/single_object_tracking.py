from pathlib import Path
from collections import OrderedDict
import copy
from collections import defaultdict
import json
import random
import mmcv

from tqdm import tqdm 

from base_dataset import BaseDataset
from prompt.utils import *


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


class youtubevis2019_sot(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/ytvis_2019/train/JPEGImages",
        "anno_path": "/path/to/lvlm_evaluation/data/ytvis_2019/train.json",
        "sampling_num": 100,
        "url": "https://www.kaggle.com/datasets/kumarujjawal123456/famous-religious-symbols",
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx",
        "sampling_range": 5,
        "sampling_ratio": 0.5
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def get_data_info(self):
        print(f"Loading {self.__class__.__name__} annotations...")
        with open(self.anno_path, "r") as f:
            anno_info = json.load(f)

        videos = copy.deepcopy(anno_info["videos"])
        annotations = anno_info["annotations"]

        video2anno = defaultdict(list)
        for i, anno in enumerate(annotations):
            video2anno[anno["video_id"]].append(i)
        
        for video in videos:
            video["annotations"] = [annotations[i] for i in video2anno[video["id"]]]

        return videos

    def convert(self):
        for video in tqdm(self.data_info):
            video_length = len(video["file_names"])

            frame_pair_list = random_frame_sampling(video_length, self.sampling_range, self.sampling_ratio)

            for frame_pair in frame_pair_list:
                frame_1_id = frame_pair[0]
                frame_2_id = frame_pair[1]

                image_1_name = video["file_names"][frame_1_id]
                image_2_name = video["file_names"][frame_2_id]
                pair_image_1 = [f"{self.image_path}/{image_1_name}", f"{self.image_path}/{image_2_name}"]

                img_width, img_height = video["width"], video["height"]
                
                pad_x0, pad_y0, img_width, img_height = get_padding_image_size(img_width, img_height)

                conversation = []
                bbox_image_1 = []
                bbox_image_2 = []

                for instance_anno in video["annotations"]:
                    bbox_1 = instance_anno["bboxes"][frame_1_id]
                    bbox_2 = instance_anno["bboxes"][frame_2_id]
                    
                    if bbox_1 != None:
                        if bbox_2 == None:
                            gpt_answer = "The object is not shown in image2."
                            bbox_image_2.append(None)
                        else:
                            x1, y1, x2, y2 = get_bbox_coords(bbox_2, img_width, img_height, pad_x0, pad_y0, mode="xywh")
                            gpt_answer = f"[{format(x1, '.3f')},{format(y1, '.3f')},{format(x2, '.3f')},{format(y2, '.3f')}]."
                            bbox_image_2.append(bbox_2)

                        x1, y1, x2, y2 = get_bbox_coords(bbox_1, img_width, img_height, pad_x0, pad_y0, mode="xywh")
                        question = f"Given the [{format(x1, '.3f')},{format(y1, '.3f')},{format(x2, '.3f')},{format(y2, '.3f')}] shown in image1, whats't the corresponding region in image2?"
                        bbox_image_1.append(bbox_1)
                        conversation.extend([
                            {"from": "human", "value": question},
                            {"from": "gpt", "value": gpt_answer}
                        ])
                
                if len(conversation) != 0:

                    image_info = self.image_dict(pair_image_1[0])

                    image_info.update(
                        {
                            "original_image_path": pair_image_1,
                            "bbox_image_1": bbox_image_1,
                            "bbox_image_2": bbox_image_2
                        }
                    )

                    if bbox_image_2[0] == None:
                        continue

                    self.images_info.append(image_info)
    
    def parse_images_info(self):
        self.data_info = self.get_data_info()
        self.images_info = list()
        self.category_space = []

        self.convert()
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import json
        import mmcv

        num_choices = 4
        width, height = image_info["width"], image_info["height"]

        question = f"Here is an object (marked as RED box) in the Frame 1. Please give the coordinations of this object in the Frame 2. Provide the output for the object in the format [x, y, w, h]. This format represents the bounding box, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as its width and height. Note that the width of the input RGB image is {width} and the height is {height}."

        bbox = image_info["bbox_image_1"][0]
        gt_bbox = image_info["bbox_image_2"][0]

        BaseDataset.exist_or_mkdir(save_image_path)
        bbox_list = [np.array([np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])])]
        new_image_path = str(Path(save_image_path) / BaseDataset.new_image_name())
        mmcv.imshow_bboxes(image_info["original_image_path"][0], bbox_list, show=False, out_file=new_image_path, thickness=2, colors="red")

        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_two_images(new_image_path, image_info["original_image_path"][1], "Image 1", "Image 2", merge_image_path, dpi=200)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_bounding_box_from_single_bbox(gt_bbox, width, height, num_choices - 1)
                    
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt_bbox,
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


class ovis_sot(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/ytvis_2019/train/JPEGImages",
        "anno_path": "/path/to/lvlm_evaluation/data/ytvis_2019/train.json",
        "sampling_num": 100,
        "url": "https://www.kaggle.com/datasets/kumarujjawal123456/famous-religious-symbols",
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx",
        "sampling_range": 5,
        "sampling_ratio": 0.5
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def get_data_info(self):
        print(f"Loading {self.__class__.__name__} annotations...")
        with open(self.anno_path, "r") as f:
            anno_info = json.load(f)

        videos = copy.deepcopy(anno_info["videos"])
        annotations = anno_info["annotations"]

        video2anno = defaultdict(list)
        for i, anno in enumerate(annotations):
            video2anno[anno["video_id"]].append(i)
        
        for video in videos:
            video["annotations"] = [annotations[i] for i in video2anno[video["id"]]]

        return videos

    def convert(self):
        for video in tqdm(self.data_info):
            video_length = len(video["file_names"])

            frame_pair_list = random_frame_sampling(video_length, self.sampling_range, self.sampling_ratio)

            for frame_pair in frame_pair_list:
                frame_1_id = frame_pair[0]
                frame_2_id = frame_pair[1]

                image_1_name = video["file_names"][frame_1_id]
                image_2_name = video["file_names"][frame_2_id]
                pair_image_1 = [f"{self.image_path}/{image_1_name}", f"{self.image_path}/{image_2_name}"]

                img_width, img_height = video["width"], video["height"]
                
                pad_x0, pad_y0, img_width, img_height = get_padding_image_size(img_width, img_height)

                conversation = []
                bbox_image_1 = []
                bbox_image_2 = []

                for instance_anno in video["annotations"]:
                    bbox_1 = instance_anno["bboxes"][frame_1_id]
                    bbox_2 = instance_anno["bboxes"][frame_2_id]
                    
                    if bbox_1 != None:
                        if bbox_2 == None:
                            gpt_answer = "The object is not shown in image2."
                            bbox_image_2.append(None)
                        else:
                            x1, y1, x2, y2 = get_bbox_coords(bbox_2, img_width, img_height, pad_x0, pad_y0, mode="xywh")
                            gpt_answer = f"[{format(x1, '.3f')},{format(y1, '.3f')},{format(x2, '.3f')},{format(y2, '.3f')}]."
                            bbox_image_2.append(bbox_2)

                        x1, y1, x2, y2 = get_bbox_coords(bbox_1, img_width, img_height, pad_x0, pad_y0, mode="xywh")
                        question = f"Given the [{format(x1, '.3f')},{format(y1, '.3f')},{format(x2, '.3f')},{format(y2, '.3f')}] shown in image1, whats't the corresponding region in image2?"
                        bbox_image_1.append(bbox_1)
                        conversation.extend([
                            {"from": "human", "value": question},
                            {"from": "gpt", "value": gpt_answer}
                        ])
                
                if len(conversation) != 0:

                    image_info = self.image_dict(pair_image_1[0])

                    image_info.update(
                        {
                            "original_image_path": pair_image_1,
                            "bbox_image_1": bbox_image_1,
                            "bbox_image_2": bbox_image_2
                        }
                    )
                    if bbox_image_2[0] == None:
                        continue

                    self.images_info.append(image_info)
    
    def parse_images_info(self):
        self.data_info = self.get_data_info()
        self.images_info = list()
        self.category_space = []

        self.convert()
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import json
        import mmcv

        num_choices = 4
        width, height = image_info["width"], image_info["height"]

        question = f"Here is an object (marked as RED box) in the Frame 1. Please give the coordinations of this object in the Frame 2. Provide the output for the object in the format [x, y, w, h]. This format represents the bounding box, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as its width and height. Note that the width of the input RGB image is {width} and the height is {height}."

        bbox = image_info["bbox_image_1"][0]
        gt_bbox = image_info["bbox_image_2"][0]

        BaseDataset.exist_or_mkdir(save_image_path)
        bbox_list = [np.array([np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])])]
        new_image_path = str(Path(save_image_path) / BaseDataset.new_image_name())
        mmcv.imshow_bboxes(image_info["original_image_path"][0], bbox_list, show=False, out_file=new_image_path, thickness=2, colors="red")

        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_two_images(new_image_path, image_info["original_image_path"][1], "Image 1", "Image 2", merge_image_path, dpi=200)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_bounding_box_from_single_bbox(gt_bbox, width, height, num_choices - 1)
                    
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt_bbox,
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



def quadrilateral_to_xyxy(coords):
    """
    将由四个角点定义的任意四边形转换为轴对齐的边界框坐标（xyxy格式）。

    :param coords: 四边形的四个角点坐标，格式为 [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: 轴对齐的边界框坐标 (xmin, ymin, xmax, ymax)
    """
    x_coords = coords[0::2]  # 提取所有x坐标
    y_coords = coords[1::2]  # 提取所有y坐标

    xmin = min(x_coords)
    ymin = min(y_coords)
    xmax = max(x_coords)
    ymax = max(y_coords)

    return xmin, ymin, xmax, ymax


class vot2018(BaseDataset):
    DATA_METAINFO = {
        "image_path": "s3://odl-dsdl/VOT2018/prepared/",
        "annotation_path": "s3://odl-dsdl/VOT2018/prepared/",
        "save_image_path": "/path/to/lvlm_evaluation/data_process/taskonomy_evaluation_data/cross_image_matching/single_object_tracking/images",
        "sampling_range": 5,
        "sampling_ratio": 0.5,
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx",
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    # def get_data_info(self):
    #     print(f"Loading {self.__class__.__name__} annotations...")
    #     with open(self.anno_path, "r") as f:
    #         anno_info = json.load(f)

    #     videos = copy.deepcopy(anno_info["videos"])
    #     annotations = anno_info["annotations"]

    #     video2anno = defaultdict(list)
    #     for i, anno in enumerate(annotations):
    #         video2anno[anno["video_id"]].append(i)
        
    #     for video in videos:
    #         video["annotations"] = [annotations[i] for i in video2anno[video["id"]]]

    #     return videos
    
    def get_data_info(self):
        print(f"Loading {self.__class__.__name__} annotations...")
        file_client = mmcv.FileClient(backend='petrel', conf_path="./petreloss.conf")
        video_names = []

        for video_name in file_client.list_dir_or_file(self.image_path):
            video_names.append(video_name)
        
        data_info = []

        for video_name in tqdm(video_names):
            # 迭代sub_video_names
            # pass
            # current_video_path = os.path.join(self.image_path, video_name)
            # sub_video_names = []
            # for sub_video_name in file_client.list_dir_or_file(current_video_path):
            #     sub_video_names.append(sub_video_name)
            
            # for sub_video_name in sub_video_names:
            """
            video_length
            file_names
            width
            height
            annotations: [
                [],
                [],
                []
            ]
            """
            video = {}

            occlusion_file = os.path.join(self.annotation_path, video_name, "occlusion.tag")
            occlusion_info = file_client.get_text(occlusion_file)
            occlusion_info_list = occlusion_info.strip().split('\n')
            num_occlusion = sum([int(v) for v in occlusion_info_list])
            if num_occlusion != 0:
                print(123)

            gt_file = os.path.join(self.annotation_path, video_name, "groundtruth.txt")
            video_img_path = os.path.join(self.image_path, video_name)

            anno_info = file_client.get_text(gt_file)
            anno_info_list = anno_info.strip().split('\n')
            file_names = []

            for frame_name in file_client.list_dir_or_file(video_img_path):
                if frame_name[-4:] == '.jpg':
                    file_names.append(os.path.join(video_name, frame_name))
            file_names.sort()
            video["file_names"] = file_names
            video_length = len(file_names)
            assert video_length < 100000, "视频太长了"
            video["video_length"] = video_length

            # get width and height
            img_url = os.path.join(self.image_path, file_names[0])
            img_bytes = file_client.get(img_url)
            image = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            video["height"], video["width"] = image.shape[:2]
            
            annotations = defaultdict(list)
            for frame_id, anno_line in enumerate(anno_info_list):
                # [x1, y1, x2, y2, x3, y3, x4, y4]
                # 旋转坐标
                coord_list = [float(v) for v in anno_line.strip().split(',')]
                x1, y1, x2, y2 = quadrilateral_to_xyxy(coord_list)

                if len(annotations[0]) == 0:
                    annotations[0] = [None for _ in range(100000)]
                
                if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:  # 目标消失
                    continue
                else:
                    annotations[0][frame_id] = [x1, y1, x2, y2]
            
            new_annotations = []

            for _, annotation in annotations.items():
                new_annotations.append(annotation[:video_length])
            
            video["annotations"] = new_annotations
            
            data_info.append(video)

        return data_info

    def convert(self):
        for video in tqdm(self.data_info):
            video_length = len(video["file_names"])

            frame_pair_list = random_frame_sampling(video_length, self.sampling_range, self.sampling_ratio)

            for frame_pair in frame_pair_list:
                frame_1_id = frame_pair[0]
                frame_2_id = frame_pair[1]

                image_1_name = video["file_names"][frame_1_id]
                image_2_name = video["file_names"][frame_2_id]
                pair_image_1 = [f"{self.image_path}/{image_1_name}", f"{self.image_path}/{image_2_name}"]

                img_width, img_height = video["width"], video["height"]
                
                pad_x0, pad_y0, img_width, img_height = get_padding_image_size(img_width, img_height)

                conversation = []
                bbox_image_1 = []
                bbox_image_2 = []

                for instance_anno in video["annotations"]:
                    bbox_1 = instance_anno["bboxes"][frame_1_id]
                    bbox_2 = instance_anno["bboxes"][frame_2_id]
                    
                    if bbox_1 != None:
                        if bbox_2 == None:
                            gpt_answer = "The object is not shown in image2."
                            bbox_image_2.append(None)
                        else:
                            x1, y1, x2, y2 = get_bbox_coords(bbox_2, img_width, img_height, pad_x0, pad_y0, mode="xywh")
                            gpt_answer = f"[{format(x1, '.3f')},{format(y1, '.3f')},{format(x2, '.3f')},{format(y2, '.3f')}]."
                            bbox_image_2.append(bbox_2)

                        x1, y1, x2, y2 = get_bbox_coords(bbox_1, img_width, img_height, pad_x0, pad_y0, mode="xywh")
                        question = f"Given the [{format(x1, '.3f')},{format(y1, '.3f')},{format(x2, '.3f')},{format(y2, '.3f')}] shown in image1, whats't the corresponding region in image2?"
                        bbox_image_1.append(bbox_1)
                        conversation.extend([
                            {"from": "human", "value": question},
                            {"from": "gpt", "value": gpt_answer}
                        ])
                
                if len(conversation) != 0:

                    image_info = self.image_dict(pair_image_1[0])

                    image_info.update(
                        {
                            "original_image_path": pair_image_1,
                            "bbox_image_1": bbox_image_1,
                            "bbox_image_2": bbox_image_2
                        }
                    )

                    self.images_info.append(image_info)
    
    def parse_images_info(self):
        self.data_info = self.get_data_info()
        self.images_info = list()
        self.category_space = []

        self.convert()
