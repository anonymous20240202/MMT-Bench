import os 
from pathlib import Path

import mmcv
from base_dataset import BaseDataset


class poetry(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/metainfo/poetry_data.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image", ],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        metadata_info = mmcv.load(self.anno_path)

        for image_info in metadata_info["images"]:
            image_info["source"] = self.dataset_name
            image_info["original_image_path"] = image_info["image_path"]

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import mmcv
        import numpy as np
        import json
        import random
        import json
        import mmcv
        from openai import OpenAI
        from prompt.utils import encode_image_to_base64
        num_choices = 4

        question = "Please write a poem based on the image."
        caption = image_info["poem"]

        gpt4v_prompt = f"Based on the image and the correct poetry I provided, generate three incorrect poetry options that are misleading. The generated error verse should be related to the picture and confusing (Similar to correct poem but wrong). \nCorrect poem: {caption}"
        chatgpt_prompt = "Your task is to extract options from the provided text and return them in JSON format, encapsulated within a Python dictionary. This dictionary should contain only one key, options, whose corresponding value is a list, with each element of the list being an option. Please do not include option symbols such as A, B, C in the list elements."

        # Path to your image
        image_path = image_info["original_image_path"]

        from PIL import Image
        # Getting the base64 string
        base64_image = encode_image_to_base64(Image.open(image_path), 768)

        openai_client = OpenAI()
        response = openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": gpt4v_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "image_detail": "low"
                        }
                    },
                ],
                }
            ],
            max_tokens=300,
        )

        options = BaseDataset.openai_generate(chatgpt_prompt, response.choices[0].message.content)["options"]

        i = 0
        while i <= 10:
            try:
                options = BaseDataset.openai_generate(chatgpt_prompt, response.choices[0].message.content)["options"]
                qa_json = {
                    "wrong_choices_list": options,
                    "question": question,
                    "num_wrong_choices": len(options),
                    "gt": caption,
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json
