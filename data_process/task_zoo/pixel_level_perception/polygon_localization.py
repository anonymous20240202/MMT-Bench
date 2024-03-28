from pathlib import Path

from base_dataset import BaseDataset

from tqdm import tqdm

import numpy as np
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np


def generate_incorrect_polygons_with_labels(correct_polygons, labels, image_width, image_height, num_options=3):
    """
    Generate incorrect polygons for a given list of correct polygons, along with their labels.
    
    :param correct_polygons: List of correct polygons in the format [x1, y1, x2, y2, ..., xn, yn]
    :param labels: List of labels corresponding to each polygon
    :param image_width: Width of the image
    :param image_height: Height of the image
    :param num_options: Number of incorrect options to generate
    :return: List of tuples containing incorrect polygons and their labels
    """

    incorrect_options = []

    for _ in range(num_options):
        # Copy the correct polygons and their labels to start modifications
        modified_polygons = [polygon.copy() for polygon in correct_polygons]
        modified_labels = labels.copy()

        for idx, polygon in enumerate(modified_polygons):
            # Randomly decide the type of modification
            modification_type = random.choice(["resize", "reposition", "None"])

            if modification_type == "resize":
                # Resize the polygon by a random factor
                resize_factor = random.uniform(0.8, 1.2)
                polygon = [max(0, min(int(coord * resize_factor), image_width if i % 2 == 0 else image_height))
                           for i, coord in enumerate(polygon)]
                modified_polygons[idx] = polygon

            elif modification_type == "reposition":
                # Reposition the polygon by a small random offset
                offset_x = random.randint(-20, 20)
                offset_y = random.randint(-20, 20)
                polygon = [max(0, min(coord + (offset_x if i % 2 == 0 else offset_y), image_width if i % 2 == 0 else image_height))
                           for i, coord in enumerate(polygon)]
                modified_polygons[idx] = polygon

            elif modification_type == "None":
                pass

        # Perform additional modifications with randomness
        additional_modifications = ["add", "remove", "duplicate"]
        random.shuffle(additional_modifications)

        for mod in additional_modifications:
            if mod == "add" and random.choice([True, False]):
                # Add a new random polygon with a random label
                new_polygon = [random.randint(0, image_width), random.randint(0, image_height)] * (len(correct_polygons[0]) // 2)
                modified_polygons.append(new_polygon)
                modified_labels.append(random.choice(labels))

            elif mod == "remove" and modified_polygons and random.choice([True, False]):
                # Remove a random polygon and its label

                if len(modified_polygons) == 1:
                    continue
                remove_index = random.choice(range(len(modified_polygons)))
                del modified_polygons[remove_index]
                del modified_labels[remove_index]

            elif mod == "duplicate" and random.choice([True, False]):
                # Duplicate a polygon with a slight offset
                duplicate_index = random.choice(range(len(correct_polygons)))
                duplicated_polygon = correct_polygons[duplicate_index].copy()
                offset_x = random.randint(-20, 20)
                offset_y = random.randint(-20, 20)
                duplicated_polygon = [max(0, min(coord + (offset_x if i % 2 == 0 else offset_y), image_width if i % 2 == 0 else image_height))
                                      for i, coord in enumerate(duplicated_polygon)]
                modified_polygons.append(duplicated_polygon)
                modified_labels.append(labels[duplicate_index])

        # Add the modified list of polygons and labels to the incorrect options
        incorrect_options.append((modified_polygons, modified_labels))

    return incorrect_options

def sample_points_from_polygon(polygon, num_samples=24):
    def calculate_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # 计算多边形每条边的长度
    edges = []
    total_perimeter = 0
    for i in range(0, len(polygon), 2):
        p1 = (polygon[i], polygon[i + 1])
        p2 = (polygon[(i + 2) % len(polygon)], polygon[(i + 3) % len(polygon)])
        edge_length = calculate_distance(p1, p2)
        edges.append((p1, p2, edge_length))
        total_perimeter += edge_length

    # 按比例分配采样点数
    samples = []
    for p1, p2, edge_length in edges:
        num_edge_samples = round((edge_length / total_perimeter) * num_samples)
        for i in range(num_edge_samples):
            alpha = i / max(num_edge_samples - 1, 1)  # 避免除以零
            sample_point = (p1[0] * (1 - alpha) + p2[0] * alpha, p1[1] * (1 - alpha) + p2[1] * alpha)
            samples.append(sample_point)

    # 调整采样点总数以满足N的要求
    while len(samples) > num_samples:
        samples.pop(random.randint(0, len(samples) - 1))
    while len(samples) < num_samples:
        edge = random.choice(edges)
        samples.append(((edge[0][0] + edge[1][0]) / 2, (edge[0][1] + edge[1][1]) / 2))

    return samples

def plot_polygon_on_image(image_path, polygon):
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB

    # 确保多边形的第一个和最后一个点是相同的，以闭合多边形
    if polygon[:2] != polygon[-2:]:
        polygon.extend(polygon[:2])

    # 将平面列表转换为2D数组，用于绘图
    polygon = np.array(polygon).reshape(-1, 2)

    plt.figure()
    plt.imshow(image)
    plt.plot(polygon[:, 0], polygon[:, 1], marker='o', color='red')  # 绘制多边形的边和顶点
    plt.fill(polygon[:, 0], polygon[:, 1], alpha=0.3, color='yellow')  # 选择性地填充多边形

    plt.title('Polygon on Image')
    plt.axis('off')  # 关闭坐标轴
    # plt.show()
    plt.savefig("out.jpg")


class coco_polygon(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/coco/val2017",
        "anno_path": "/path/to/lvlm_evaluation/data/coco/annotations/instances_val2017.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import mmcv
        from collections import defaultdict

        self.images_info = list()
        self.category_space = []
        anno_data_info = mmcv.load(self.anno_path)
        
        image2anno = defaultdict(list)

        for anno_info in anno_data_info["annotations"]:
            # segmentation = anno_info["segmentation"]
            image2anno[anno_info["image_id"]].append(anno_info)

        id2category = dict()
        for cate_info in anno_data_info["categories"]:
            id2category[cate_info['id']] = cate_info["name"]
            self.category_space.append(cate_info["name"])
        
        for image_info in tqdm(anno_data_info["images"]):
            segment_list = image2anno[image_info["id"]]

            category_list = []
            segmentation_lsit = []

            flag = True
            for segment_info in segment_list:
                category = id2category[segment_info["category_id"]]
                segmentation = segment_info["segmentation"]

                if isinstance(segmentation, list):
                    if len(segmentation) > 1:
                        flag = False
                    else:
                        pass
                else:
                    flag = False


                category_list.append(category)
                segmentation_lsit.append(segmentation)
            
            if flag is False:
                continue

            if len(segmentation_lsit) >1 or len(segmentation_lsit) == 0:
                continue
            
            image_path = Path(self.image_path) / image_info["file_name"]
            
            width, height = self.get_image_width_height(image_path)
            
            image_info = self.image_dict("")
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "segmentation_list": segmentation_lsit,
                    "category": category_list
                }
            )

            self.images_info.append(image_info)

        self.dataset_info["category_space"] = self.category_space
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 2
        width = image_info["width"]
        height = image_info["height"]
        question = f"Please detect all instances of the following categories in this image: {', '.join(dataset_info['category_space'])}. For each detected object, provide the output in the format: category: [x1, y1, x2, y2, x3, y3, x4, y4]. This format represents a rotated bounding box for each object. The points [x1, y1], [x2, y2], [x3, y3], and [x4, y4] are the coordinates of the four corners of the bounding box. Note that the width of the input image is given as {width} and the height as {height}."

        polygon_list = []
        category_list = []

        for segmentation, category in zip(image_info["segmentation_list"], image_info["category"]):
            assert len(segmentation) == 1
            polygon = segmentation[0]

            sampled_polygon = sample_points_from_polygon(polygon, num_samples=24)

            new_polygon = []

            for polygon in sampled_polygon:
                new_polygon.extend([round(polygon[0]), round(polygon[1])])
            polygon_list.append(new_polygon)
            category_list.append(category)
        
        
        # output = generate_incorrect_polygons_with_labels(polygon_list, category_list, width, height, num_choices-1)


        _t = []
        for bbox, cate in zip(polygon_list, category_list):
            _t.append(f"{cate}: {bbox}")
        gt = ", ".join(_t)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_polygons_with_labels(polygon_list, category_list, width, height, num_choices-1)
                new_wrong_choices_list = []
                for wrong_choice in wrong_choices_list:
                    _t = []
                    for bbox, cate in zip(wrong_choice[0], wrong_choice[1]):
                        _t.append(f"{cate}: {bbox}")
                    
                    new_wrong_choices_list.append(", ".join(_t))
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt,
                    "question": question,
                    "wrong_choices_list": new_wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json


class youtubevis2019_polygon(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/ytvis_2019/train/JPEGImages",
        "anno_path": "/path/to/lvlm_evaluation/data/ytvis_2019/train.json",
        "sampling_num": 50,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import mmcv
        from collections import defaultdict

        self.images_info = list()
        self.category_space = []
        anno_data_info = mmcv.load(self.anno_path)
        
        video2anno = defaultdict(list)

        for anno_info in anno_data_info["annotations"]:
            # segmentation = anno_info["segmentation"]
            video2anno[anno_info["video_id"]].append(anno_info)

        id2category = dict()
        for cate_info in anno_data_info["categories"]:
            id2category[cate_info['id']] = cate_info["name"]
            self.category_space.append(cate_info["name"])
        
        for video_info in tqdm(anno_data_info["videos"]):
            segment_list = video2anno[video_info["id"]]

            for i, file_name in enumerate(video_info["file_names"]):
                image_path = Path(self.image_path) / file_name

                segmentation_list = []
                category_list = []

                for segmentation in segment_list:
                    if segmentation["segmentations"][i] == None:
                        continue
                    else:
                        category_list.append(id2category[segmentation["category_id"]])
                        segmentation_list.append(segmentation["segmentations"][i])

                if len(category_list) == 0:
                    continue
                width, height = self.get_image_width_height(image_path)
                
                image_info = self.image_dict("")
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "width": width,
                        "height": height,
                        "segmentation_list": segmentation_list,
                        "category": category_list
                    }
                )

                self.images_info.append(image_info)

        self.dataset_info["category_space"] = self.category_space


class ovis_polygon(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/OpenDataLab___OVIS/raw/train",
        "anno_path": "/path/to/lvlm_evaluation/data/OpenDataLab___OVIS/raw/annotations_train.json",
        "sampling_num": 50,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import mmcv
        from collections import defaultdict

        self.images_info = list()
        self.category_space = []
        anno_data_info = mmcv.load(self.anno_path)
        
        video2anno = defaultdict(list)

        for anno_info in anno_data_info["annotations"]:
            # segmentation = anno_info["segmentation"]
            video2anno[anno_info["video_id"]].append(anno_info)

        id2category = dict()
        for cate_info in anno_data_info["categories"]:
            id2category[cate_info['id']] = cate_info["name"]
            self.category_space.append(cate_info["name"])
        
        for video_info in anno_data_info["videos"]:
            segment_list = video2anno[video_info["id"]]

            for i, file_name in enumerate(video_info["file_names"]):
                image_path = Path(self.image_path) / file_name

                segmentation_list = []
                category_list = []

                for segmentation in segment_list:
                    if segmentation["segmentations"][i] == None:
                        continue
                    else:
                        category_list.append(id2category[segmentation["category_id"]])
                        segmentation_list.append(segmentation["segmentations"][i])

                if len(category_list) == 0:
                    continue
                width, height = self.get_image_width_height(image_path)

                if len(segmentation_list) > 1:
                    pass
                
                image_info = self.image_dict("")
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "width": width,
                        "height": height,
                        "segmentation_list": segmentation_list,
                        "category": category_list
                    }
                )

                self.images_info.append(image_info)

        self.dataset_info["category_space"] = self.category_space
