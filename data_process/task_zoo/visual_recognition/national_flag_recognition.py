from pathlib import Path
import json

from iso3166 import countries

from base_dataset import BaseDataset


class country_flag(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/country_flag",
        "sampling_num": 200,
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        country_code_list = []
        self.category_space = []
        for country in Path(self.image_path).iterdir():
            country_code = country.stem
            country_code_list.append(country_code)
            try:
                country_name = countries.get(country_code.upper()).name
            except:
                continue

            image_info = self.image_dict(str(country))
            image_info.update(
                {
                    "original_image_path": str(country),
                    "category": country_name
                }
            )
            if '\n' in country_name:
                continue
            self.category_space.append(country_name)

            self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = self.category_space
    
    @staticmethod
    def generate_qa(image_info, dataset_info, _):
        import json
        import mmcv

        num_choices = 4
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": "Which country does the flag in the picture belong to?",
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "America",
                "question": "Which country does the flag in the picture belong to?",
                "wrong_choices_list": ["China", "Australia", "England"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["category"],
                "question": "Which country does the flag in the picture belong to?",
            }
        }

        user_prompt = json.dumps(input_json)

        while True:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json, question="Which country does the flag in the picture belong to?")
                break
            except:
                pass

        return qa_json
    