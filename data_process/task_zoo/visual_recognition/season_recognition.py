from pathlib import Path

from base_dataset import BaseDataset


class image_season_recognition(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/recognition_season/train",
        "sampling_num": 200,
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.category_space = [
            "spring", "summer", "fall", "winter"
        ]
        self.dataset_info["category_space"] = self.category_space

    
    def parse_images_info(self):
        self.images_info = list()

        for season_path in Path(self.image_path).iterdir():
            season_category = season_path.name
            for image_path in season_path.iterdir():
                original_image_path = str(image_path)

                image_info = self.image_dict
                image_info.update(
                    {
                        "category": season_category,
                        "original_image_path": original_image_path
                    }
                )
                
                self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv
        import copy

        num_choices = 4
        question = "What is the season shown in the picture?"

        i = 0
        while i <= 10:
            try:
                category_space = copy.deepcopy(dataset_info["category_space"])
                category_space.remove(image_info["category"])
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": image_info["category"],
                    "question": question,
                    "wrong_choices_list": category_space
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json
                