from pathlib import Path
from collections import OrderedDict

import mmcv

from base_dataset import BaseDataset


class Kitchen_World(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/taskonomy_data/embodied_ai/navigation/metadata_info.json",
        "sampling_num": 200,
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "Kitchen World creates data in the kitchen environment, inlcuding various planning tasks as the role of kitchen assistant such as heat the potato."
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)


        for image_info in anno_info["images"]:
            image_info["original_image_path"] = image_info["image_path"].replace("lustre", "petrelfs")
            
            image_info["source"] = self.dataset_name

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = image_info["question"]
        answer = image_info["gt_answer"]

        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/general_prompt.txt'))

        input_json = {
            "question": question,
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": answer,
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
