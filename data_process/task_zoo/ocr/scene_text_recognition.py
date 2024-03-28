from pathlib import Path

import struct
import numpy as np
import copy

import mmcv

from base_dataset import BaseDataset


class ICDAR2013(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/samples/scene text recognition/metadata_info_new.json",
        "sampling_num": 100,
        "visual_input_component": ["text-rich_image", ],
        "dataset_description": "The ICDAR 2013 dataset is the standard benchmark dataset for evaluating near-horizontal text detection.",
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)
        for image_info in anno_info["images"]:

            if image_info["source"] != "ICDAR2013":
                continue
            
            for bbox, text in zip(image_info["bounding_box_coordinates"], image_info["txt"]):  # xyxy
                pass
                _image_info = copy.deepcopy(image_info)

                image_info["source"] = self.dataset_name
                _image_info["bbox"] = bbox
                _image_info["text"] = text
                _image_info["original_image_path"] = _image_info["image_name"]

                self.images_info.append(_image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import mmcv
        import numpy as np
        import json
        num_choices = 4
        question = "What is the text content shown in the region (marked as RED bounding box)?"

        BaseDataset.exist_or_mkdir(save_image_path)
        bbox_list = [np.array([image_info['bbox']])]
        new_image_path = str(Path(save_image_path) / BaseDataset.new_image_name())
        mmcv.imshow_bboxes(image_info["original_image_path"], bbox_list, show=False, out_file=new_image_path, thickness=2, colors='red')


        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/general_prompt.txt'))

        input_json = {
            "question": question,
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "A133",
                "question": question,
                "wrong_choices_list": ["Clacton", "intelligent", "GROUP"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["text"],
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
        qa_json["marked_image_path"] = new_image_path

        return qa_json


class IIIT5K(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/samples/scene text recognition/metadata_info_new.json",
        "sampling_num": 100,
        "visual_input_component": ["text-rich_image", ],
        "dataset_description": "The IIIT5K dataset [31] contains 5,000 text instance images: 2,000 for training and 3,000 for testing. It contains words from street scenes and from originally-digital images. Every image is associated with a 50 -word lexicon and a 1,000 -word lexicon. Specifically, the lexicon consists of a ground-truth word and some randomly picked words.",
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)
        for image_info in anno_info["images"]:

            if image_info["source"] != "IIIT5K":
                continue

            image_info["original_image_path"] = image_info["image_name"]
            image_info["source"] = self.dataset_name
            image_info["text"] = image_info["label"]


            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import mmcv
        import numpy as np
        import json
        num_choices = 4
        question = "What is the text content shown in the image?"

        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/general_prompt.txt'))

        input_json = {
            "question": question,
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "A133",
                "question": question,
                "wrong_choices_list": ["Clacton", "intelligent", "GROUP"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["text"],
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
    