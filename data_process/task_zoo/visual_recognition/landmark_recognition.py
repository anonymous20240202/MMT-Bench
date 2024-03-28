from pathlib import Path

from base_dataset import BaseDataset


class landmark_internet(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/landmark/landmark",
        "sampling_num": 50,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for image_path in Path(self.image_path).iterdir():
            category = image_path.stem
            self.category_space.append(category)

            image_info = self.image_dict(str(image_path))
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "category": category
                }
            )

            self.images_info.append(image_info)

        self.dataset_info["category_space"] = self.category_space
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the name of the landmark in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Victoria Falls, Canada",
                "question": question,
                "wrong_choices_list": ["Colosseum", "Egyptian Pyramids", "Lake Bled"]
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
        