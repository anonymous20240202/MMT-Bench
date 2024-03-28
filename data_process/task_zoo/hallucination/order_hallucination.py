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



class Flickr30K_Order(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/taskonomy_data/visual_hallucination/order_hallucination/Flickr/metadata_info.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "Flickr30K Order tests the understanding of objects' relation in complex natural scenes. Given an image and a constituent relation of the form X relation Y, we test whether the model can pick the correct order."
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        
        anno_info = mmcv.load(self.anno_path)

        for image_info in anno_info["images"]:

            image_info["source"] = self.dataset_name
            image_info["original_image_path"] = image_info["image_path"].replace("lustre", "petrelfs")

            self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import json
        import mmcv

        num_choices = 4
        question = image_info["question"]

        wrong_choices_list = image_info['Options_answer']
        wrong_choices_list.remove(image_info["gt_answer"])

        qa_json = {
            "num_wrong_choices": num_choices - 1,
            "gt": image_info["gt_answer"],
            "question": question,
            "wrong_choices_list": wrong_choices_list[:num_choices - 1]
        }
        
        qa_json = BaseDataset.post_process(qa_json, question=question)

        return qa_json
