from pathlib import Path

import mmcv
from tqdm import tqdm

from base_dataset import BaseDataset


class visual_genome_sg(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/data/markllava/finetune/visual_genome",
        "anno_path": "/path/to/lvlm_evaluation/data_process/data/OpenDataLab___Visual_Genome_Dataset_V1_dot_2/raw/data/relationships.json",
        "image_data_path": "/path/to/lvlm_evaluation/data_process/data/OpenDataLab___Visual_Genome_Dataset_V1_dot_2/raw/data/image_data.json",
        "sampling_num": 50,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import os
        self.images_info = list()
        self.category_space = []
        relationship_data_list = mmcv.load(self.anno_path)

        image_data_list = mmcv.load(self.image_data_path)

        image_id2_image_path = {}
        image_id2_wh = {}
        for image_data in image_data_list:
            image_id2_image_path[image_data["image_id"]] = os.path.join(*image_data['url'].split('/')[-2:])
            image_id2_wh[image_data["image_id"]] = {"width": image_data['width'], "height": image_data['height'],}


        for relationship_data in tqdm(relationship_data_list):
            image_id = relationship_data["image_id"]
            image_path = str(Path(self.image_path) / image_id2_image_path[image_id])
            for relationship in relationship_data["relationships"]:

                category = relationship['predicate']
                self.category_space.append(category)
                subject_info = relationship["subject"]
                object_info = relationship["object"]

                bounding_box_coordinates = {
                    "subject": [subject_info["x"], subject_info["y"], subject_info["w"], subject_info["h"]],
                    "object": [object_info["x"], object_info["y"], object_info["w"], object_info["h"],]
                }

                image_info = self.image_dict(image_path=image_path, width=image_id2_wh[image_id]['width'], height=image_id2_wh[image_id]['height'])

                image_info.update(
                    {
                        "original_image_path": image_path,
                        "category": category,
                        "bounding_box_coordinates": bounding_box_coordinates
                    }
                )
                self.images_info.append(image_info)

        self.dataset_info["category_space"] = list(set(self.category_space))


class vrd_sg(BaseDataset):
    DATA_METAINFO = {
            "anno_path": "/path/to/lvlm_evaluation/data_process/data/sg/sg_dataset/json_dataset/annotations_test.json",
            "image_path": "/path/to/lvlm_evaluation/data_process/data/sg/sg_dataset/sg_test_images",
            "sampling_num": 50,
            "visual_input_component": ["natural_image"],
            "dataset_description": "xxx"}
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.category_space = mmcv.load("/path/to/lvlm_evaluation/data_process/data/sg/sg_dataset/json_dataset/predicates.json")
        self.dataset_info["category_space"] = self.category_space

    def parse_images_info(self):
        import os
        self.images_info = list()
        # self.category_space = []
        anno_info = mmcv.load(self.anno_path)

        for image_name, relationship_list in anno_info.items():
            image_path = str(Path(self.image_path) / image_name)
            for relationship in relationship_list:

                category = self.category_space[relationship['predicate']]
                # self.category_space.append(category)
                subject_info = relationship["subject"]
                object_info = relationship["object"]

                bounding_box_coordinates = {
                    "subject": subject_info['bbox'],
                    "object": object_info['bbox']
                }

                image_info = self.image_dict(image_path=image_path)

                image_info.update(
                    {
                        "original_image_path": image_path,
                        "category": category,
                        "bounding_box_coordinates": bounding_box_coordinates
                    }
                )
                self.images_info.append(image_info)


class vsr(BaseDataset):
    DATA_METAINFO = {
            "anno_path": "/path/to/lvlm_evaluation/data_process/data/vsr_zeroshot/dev.jsonl",
            "image_path": "/path/to/lvlm_evaluation/data_process/data/coco/train2017",
            "save_image_path": "/path/to/lvlm_evaluation/data_process/taskonomy_evaluation_data/relation_reasoning/scene_graph_recognition/images",
            "sampling_num": 200,
            "visual_input_component": ["natural_image"],
            "dataset_description": "xxx"}
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        import os
        import json
        import requests

        from tqdm import tqdm
        self.images_info = list()
        # self.category_space = []
        with open(self.anno_path, 'r') as file:
            for line in tqdm(file):
                # 使用 json.loads() 解析每一行的 JSON 对象
                json_object = json.loads(line)
                # 现在你可以访问 json_object 中的数据了

                # response = requests.get(json_object["image_link"])

                # # 检查响应状态码，200表示请求成功
                # if response.status_code == 200:
                #     # 获取图片的二进制数据
                #     image_data = response.content

                #     # 指定本地文件路径来保存图片
                #     local_file_path = os.path.join(self.save_image_path, json_object["image"])

                #     # 打开文件以二进制写入模式，并将图片数据写入文件
                #     with open(local_file_path, 'wb') as file:
                #         file.write(image_data)

                local_file_path = os.path.join(self.save_image_path, json_object["image"])
                image_name = json_object["image"]
                caption = json_object["caption"]
                label = json_object["label"]

                image_info = self.image_dict("")
                image_info.update(
                    {
                        "original_image_path": local_file_path,
                        "caption": caption,
                        "label": label
                    }
                )
        
                self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        num_choices = 2
        question = f"Please determine whether the spatial relationships described in the captions below are correct based on the image.\nCaption: {image_info['caption']}"

        if image_info["label"] == 0:
            gt = "Correct"
            wrong_choices_list = ["Error"]
        else:
            gt = "Error"
            wrong_choices_list = ["Correct"]
        qa_json = {
            "num_wrong_choices": num_choices - 1,
            "gt": gt,
            "question": question,
            "wrong_choices_list": wrong_choices_list
        }
        qa_json = BaseDataset.post_process(qa_json, question=question)

        return qa_json
