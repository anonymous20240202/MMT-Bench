from pathlib import Path

import mmcv

from base_dataset import BaseDataset


class mvtec_anomaly_detection(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/anomaly_detection/meta_info.json",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "MVTec is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection. It contains over 5000 high-resolution images divided into fifteen different object and texture categories. Each category comprises a set of defect-free training images and a test set of images with various kinds of defects as well as images without defects."
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)

        for image_info in anno_info["images"]:
            if image_info["source"] != "mvtec_anomaly_detection":
                continue
            image_info["source"] = self.dataset_name
            image_info["original_image_path"] = image_info["original_image_path"].replace("/home/wenbo/Documents/data/LLM_eval_dataset/", "/path/to/")

            import os
            if not os.path.exists(image_info["original_image_path"]):
                continue

            self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = f"How many defects are there in this {image_info['category']}?"
        judgment = image_info["judgment"]

        if judgment == 0:
            wrong_choices_list = [1,2,3]
        else:
            wrong_choices_list = [judgment - 1, judgment + 1, judgment + 2]

        qa_json = {
            "num_wrong_choices": num_choices - 1,
            "gt": judgment,
            "question": question,
            "wrong_choices_list": wrong_choices_list
        }

        qa_json = BaseDataset.post_process(qa_json, question)

        return qa_json


class MPDD(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/anomaly_detection/meta_info.json",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "A newly introduced dataset focuses on enhancing anomaly detection (AD) in painted metal parts fabrication. Traditional datasets typically comprise images from controlled laboratory settings with monochromatic backgrounds and centrally placed objects. However, such conditions are rarely met in actual manufacturing processes and production lines. To bridge this gap, the proposed dataset incorporates images featuring multiple objects with varying spatial orientations, positions, and distances relative to the camera. These are set against non-homogeneous backgrounds and under diverse lighting intensities, providing a more realistic and challenging environment for AD methods. Initial evaluations reveal that while certain state-of-the-art AD techniques excel on standard datasets like MVTec AD, their performance significantly differs on this more complex dataset. The findings underline the necessity of application-specific datasets for advancing and refining AD methodologies.."
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)

        for image_info in anno_info["images"]:
            if image_info["source"] != "MPDD":
                continue
            image_info["source"] = self.dataset_name
            image_info["original_image_path"] = image_info["original_image_path"].replace("/home/wenbo/Documents/data/LLM_eval_dataset/", "/path/to/")

            import os
            if not os.path.exists(image_info["original_image_path"]):
                continue
            self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = f"How many defects are there in this {image_info['category']}?"
        judgment = image_info["judgment"]

        if judgment == 0:
            wrong_choices_list = [1,2,3]
        else:
            wrong_choices_list = [judgment - 1, judgment + 1, judgment + 2]

        qa_json = {
            "num_wrong_choices": num_choices - 1,
            "gt": judgment,
            "question": question,
            "wrong_choices_list": wrong_choices_list
        }

        qa_json = BaseDataset.post_process(qa_json, question)

        return qa_json