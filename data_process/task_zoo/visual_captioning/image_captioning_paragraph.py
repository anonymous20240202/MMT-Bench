import os 
from pathlib import Path

import mmcv
from base_dataset import BaseDataset


# class ade20k_caption(BaseDataset):
#     DATA_METAINFO = {
#         "anno_path": "/path/to/metainfo/storytelling.json",
#         "sampling_num": 200,
#         "visual_input_component": ["natural_image", ],
#         "dataset_description": "xxx"
#     }
    
#     def parse_dataset_info(self):
#         super().parse_dataset_info()
    
#     def parse_images_info(self):
#         self.images_info = list()

#         metadata_info = mmcv.load(self.anno_path)

#         for image_info in metadata_info["stroies"]:
#             image_info["source"] = self.dataset_name
#             image_info["caption"] = image_info["d_values"]
#             image_info["original_image_path"] = image_info["image_path"]
#             image_info["original_image_path"] = image_info["original_image_path"].replace("/path/to/lvlm_evaluation/data_process/data/ade20k_/ade20k/", "/path/to/lvlm_evaluation/data_process/data/ade20k_/ade20k/images/training/")

#             self.images_info.append(image_info)


class paragraph_caption(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/lvlm_evaluation/data_process/data/OpenDataLab___Image_Paragraph_Captioning/raw/paragraphs_v1.json",
        "image_path": "/path/to/data/markllava/finetune/visual_genome",
        "sampling_num": 200,
        "visual_input_component": ["natural_image", ],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        metadata_info = mmcv.load(self.anno_path)

        for image_info in metadata_info:
            image_path = image_info["url"].replace("https://cs.stanford.edu/people/rak248", self.image_path)
            image_info["source"] = self.dataset_name
            image_info["original_image_path"] = image_path

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

        # question = image_info["question"]
        # gt = image_info["gt_answer"]
        question = "Describe this image in one paragraph."
        caption = image_info["paragraph"]

        example_dict = {
            "ground_truth_caption": caption,
            "generated_wrong_caption": [
                "At an exclusive outdoor culinary event, a gastronome displays a meticulously assembled gourmet sausage creation, featuring a blend of rare meats and infused with an array of international spices, topped with a delicate arrangement of hand-picked, heirloom vegetable garnishes,",
                "During a bustling urban food festival, a local chef exhibits a masterfully constructed fusion hot dog, boasting an innovative sausage blend of grass-fed beef and wild boar, lavishly adorned with a colorful mélange of organic, sun-ripened heirloom tomatoes",
                "At a sustainable living fair, an eco-conscious food artist presents a plant-based, cruelty-free 'not-dog', a marvel of modern cuisine designed to replicate the classic flavors of a hot dog, featuring a protein-rich, soy-based sausage substitute, ",
            ]
        }

        # example_dict_string = json.dumps(example_dict)

        # gpt4v_prompt = f"Based on the image and the correct caption I provided, generate three incorrect detail captions that are misleading. The generated paragraph has a similar structure (Similar in length) to GT and is related to the image.NOTE: The generated paragraph has a similar structure (Similar in length) to GT. Please modify the GT caption based on the picture and make three wrong caption. \nGround Truth Caption: {caption} \n Your should return in JSON format, encapsulated within a Python dictionary. This dictionary should contain only one key, options, whose corresponding value is a list, with each element of the list being an option. Please do not include option symbols such as A, B, C in the list elements."
        # chatgpt_prompt = "Your task is to extract options from the provided text and return them in JSON format, encapsulated within a Python dictionary. This dictionary should contain only one key, options, whose corresponding value is a list, with each element of the list being an option. Please do not include option symbols such as A, B, C in the list elements."

        # # Path to your image
        # image_path = image_info["original_image_path"]

        # from PIL import Image
        # # Getting the base64 string
        # base64_image = encode_image_to_base64(Image.open(image_path), 768)

        # openai_client = OpenAI()
        # # response = openai_client.chat.completions.create(
        # #     model="gpt-4-vision-preview",
        # #     messages=[
        # #         {
        # #         "role": "user",
        # #         "content": [
        # #             {"type": "text", "text": gpt4v_prompt},
        # #             {
        # #                 "type": "image_url",
        # #                 "image_url": {
        # #                     "url": f"data:image/jpeg;base64,{base64_image}",
        # #                     "image_detail": "low"
        # #                 }
        # #             },
        # #         ],
        # #         }
        # #     ],
        # #     max_tokens=300,
        # # )

        

        # i = 0
        # while i <= 10:
        #     try:
        #         response = openai_client.chat.completions.create(
        #             model="gpt-4-vision-preview",
        #             messages=[
        #                 {
        #                 "role": "user",
        #                 "content": [
        #                     {"type": "text", "text": gpt4v_prompt},
        #                     {
        #                         "type": "image_url",
        #                         "image_url": {
        #                             "url": f"data:image/jpeg;base64,{base64_image}",
        #                             "image_detail": "low"
        #                         }
        #                     },
        #                 ],
        #                 }
        #             ],
        #             max_tokens=300,
        #         )
        #         options = json.loads(response.choices[0].message.content.replace("```python\n", "").replace("```", "").replace("\n", ""))["options"]
        #         qa_json = {
        #             "wrong_choices_list": options,
        #             "question": question,
        #             "num_wrong_choices": len(options),
        #             "gt": caption,
        #         }
        #         qa_json = BaseDataset.post_process(qa_json, question=question)
        #         break
        #     except:
        #         i += 1


        # caption = image_info["caption"]
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
                qa_json.update(image_info)
                break
            except:
                i += 1

        return qa_json
        # return qa_json
