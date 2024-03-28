from pathlib import Path
from collections import OrderedDict

from base_dataset import BaseDataset


class astronomical_internet(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/star",
        "sampling_num": 50,
        "url": "internet",
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        self.category_lsit = []
        for image_path in Path(self.image_path).iterdir():
            category = image_path.stem.split("_")[0]
            self.category_lsit.append(category)
            # for image_path in artist_path.iterdir():

            image_info = self.image_dict(str(image_path))
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "catgeory": category
                }
            )
            self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = list(set(self.category_lsit))

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the name of astronomical object shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Hale-Bopp",
                "question": question,
                "wrong_choices_list": ["Cetus A", "Milky Way", "Venus"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["catgeory"],
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)

        while True:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                pass

        return qa_json