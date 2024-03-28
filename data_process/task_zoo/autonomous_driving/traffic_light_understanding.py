from pathlib import Path
from collections import OrderedDict

import mmcv

from base_dataset import BaseDataset


# class LISA_TL(BaseDataset):
#     DATA_METAINFO = {
#         "anno_path": "/path/to/taskonomy_data/autonomous_driving/traffic_light_understanding/LISA-TL/metadata_info.json",
#         "sampling_num": 100,
#         "visual_input_component": ["natural_image"],
#         "dataset_description": "LISA Traffic Light Dataset is a dataset commonly used for research and development in the field of computer vision and machine learning, particularly in the context of traffic light detection and recognition. The dataset includes annotated images that contain various traffic scenes, with a specific focus on traffic lights."
#     }
    
#     def parse_dataset_info(self):
#         super().parse_dataset_info()
    
#     def parse_images_info(self):
#         self.images_info = list()

#         anno_info = mmcv.load(self.anno_path)

#         for image_info in anno

#         pass


class S2TLD(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/taskonomy_data/autonomous_driving/traffic_light_understanding/S2TLD/metadata_info.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "S2TLD is a traffic light dataset, containing 5 categories (include red, yellow, green, off and wait on) of 1,4130 instances. The scenes cover a decent variety of road scenes."
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)

        for image_info in anno_info["images"]:
            image_info["original_image_path"] = image_info["image_path"].replace("/mnt/lustre/share_data/shaowenqi/taskonomy_data/autonomous_driving/traffic_light_understanding/S2TLD/./S2TLD/Annotations",
                                                                                 "/path/to/taskonomy_data/autonomous_driving/traffic_light_understanding/S2TLD/S2TLD/JPEGImages").replace(":", "_")
            
            if len(image_info["image_label"]) != 1:
                continue
            category = image_info["image_label"][0]

            category = "wait on" if category == "wait_on" else category

            image_info["category"] = category
            image_info["source"] = self.dataset_name
            self.images_info.append(image_info)

        self.dataset_info["category_space"] = anno_info["dataset_list"][0]["category_space"]
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = image_info["question"]
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "yellow",
                "question": question,
                "wrong_choices_list": ["red", "green", "off"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["category"],
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)

        i = 0
        while i <= 10:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json
