import os 
from pathlib import Path

import mmcv
import numpy as np
from base_dataset import BaseDataset

from tqdm import tqdm
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import json
# from prompt.utils import generate_incorrect_bounding_boxes_with_descriptions


def cat_imgs(img_list, save_image_path):
    os.makedirs(save_image_path, exist_ok=True)
    sample_frame_info = dict()
    merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())
    sample_frame_info[f'merge_image_path'] = merge_image_path
    try:
        concat_imgs = cv2.hconcat([cv2.imread(img_path) for img_path in img_list])
        cv2.imwrite(merge_image_path, concat_imgs)
    except:
        # 创建图形和子图
        fig, axes = plt.subplots(1, len(img_list), figsize=(15, 5))
        imgs = [mpimg.imread(img_path) for img_path in img_list]
        # 在子图上显示图片和标签
        for ax, img in zip(axes, imgs):
            ax.imshow(img)
            ax.axis('off')  # 关闭坐标轴
        # 调整子图之间的间隔
        plt.subplots_adjust(wspace=0.1)

        # 保存图形到指定路径
        plt.savefig(merge_image_path, bbox_inches='tight', dpi=300)

    return sample_frame_info


class SSID(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/metainfo/multi-image_ssid.json",
        "dataset_description" : "for sequential five pictures, one text is generated",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()
        anno_info = mmcv.load(self.anno_path)
        for image_info in anno_info["images_set"]:
            image_info["source"] = self.dataset_name
            image_info["original_image_path"] = list()
            for item in image_info["images"]:
                image_info["original_image_path"].append(item["image_path"])
            image_info["caption"] = image_info["storytext"]
            del image_info["images"]
            del image_info["storytext"]
            self.images_info.append(image_info)



    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        
        num_choices = 4
        image_list = image_info["original_image_path"]
        images_dict = cat_imgs(image_list, save_image_path)

        question = "Describe this set of images briefly."
        caption = image_info["caption"]
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/general_prompt.txt'))

        input_json = {
            "question": question,
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": caption,
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)

        i = 0
        while i <= 10:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json, question=question)
                qa_json.update(images_dict)
                break
            except:
                i += 1

        return qa_json




class VIST(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/metainfo/multi-image_vist.json",
        "dataset_description" : "The dataset is VIST-SII(story in sequence, every five pictures are mapped to one story",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()
        anno_info = mmcv.load(self.anno_path)
        for image_info in anno_info["images_set"]:
            image_info["source"] = self.dataset_name
            image_info["original_image_path"] = list()
            for item in image_info["image_list"]:
                image_info["original_image_path"].append(item["image_path"])
            del image_info["image_list"]
            image_info['caption'] = ''.join(image_info['caption'])
            self.images_info.append(image_info)
    
    # @staticmethod
    # def generate_qa(image_info, dataset_info, save_image_path):
    #     import mmcv
    #     import numpy as np
    #     import json
    #     import random
    #     import json
    #     import mmcv
    #     from openai import OpenAI
    #     from prompt.utils import encode_image_to_base64
    #     num_choices = 4

    #     # question = image_info["question"]
    #     # gt = image_info["gt_answer"]
    #     question = "Describe this set of images briefly."
    #     caption = image_info["caption"]

    #     gpt4v_prompt = f"Based on the set of images and the correct caption I provided, generate three incorrect captions that are misleading.\nCorrect Caption: {caption}"
    #     chatgpt_prompt = "Your task is to extract options from the provided text and return them in JSON format, encapsulated within a Python dictionary. This dictionary should contain only one key, options, whose corresponding value is a list, with each element of the list being an option. Please do not include option symbols such as A, B, C in the list elements."

    #     # Path to your image
    #     image_list = image_info["original_image_path"]
    #     images_dict = cat_imgs(image_list, save_image_path)
        
    #     from PIL import Image
    #     # Getting the base64 string
    #     base64_image = encode_image_to_base64(Image.open(images_dict["merge_image_path"]), 768)

    #     openai_client = OpenAI()
    #     response = openai_client.chat.completions.create(
    #         model="gpt-4-vision-preview",
    #         messages=[
    #             {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": gpt4v_prompt},
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{base64_image}",
    #                         "image_detail": "low"
    #                     }
    #                 },
    #             ],
    #             }
    #         ],
    #         max_tokens=300,
    #     )

    #     i = 0
    #     while i <= 10:
    #         try:
    #             options = BaseDataset.openai_generate(chatgpt_prompt, response.choices[0].message.content)["options"]
    #             qa_json = {
    #                 "wrong_choices_list": options,
    #                 "question": question,
    #                 "num_wrong_choices": len(options),
    #                 "gt": caption,
    #             }
    #             qa_json.update(images_dict)
    #             qa_json = BaseDataset.post_process(qa_json, question=question)
    #             break
    #         except:
    #             i += 1

    #     return qa_json
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        
        num_choices = 4
        image_list = image_info["original_image_path"]
        images_dict = cat_imgs(image_list, save_image_path)

        question = "Describe this set of images briefly."
        caption = image_info["caption"]
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/general_prompt.txt'))

        input_json = {
            "question": question,
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": caption,
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)

        i = 0
        while i <= 10:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json, question=question)
                qa_json.update(images_dict)
                break
            except:
                i += 1

        return qa_json

