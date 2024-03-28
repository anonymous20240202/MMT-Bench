from pathlib import Path

from base_dataset import BaseDataset


class indoor_scene_recognition(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/scene_recognition/Images",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        # self.category_space = [
        #     "spring", "summer", "fall", "winter"
        # ]
        # self.dataset_info["category_space"] = self.category_space

    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        for scene_path in Path(self.image_path).iterdir():
            scene_category = scene_path.name
            self.category_space.append(scene_category)
            for image_path in scene_path.iterdir():
                original_image_path = str(image_path)

                image_info = self.image_dict(str(original_image_path))
                image_info.update(
                    {
                        "category": scene_category,
                        "original_image_path": original_image_path
                    }
                )
                
                self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = self.category_space

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What category of scene is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "gameroom",
                "question": question,
                "wrong_choices_list": ["mall", "tv_studio", "office"]
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


class places365(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/scene_recognition/OpenDataLab___Places365/raw/places/data_256",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        for a_path in Path(self.image_path).iterdir():
            for scene_category_path in a_path.iterdir():
                scene_category = scene_category_path.name
                self.category_space.append(scene_category)

                for image_path in scene_category_path.iterdir():
                    image_info = self.image_dict(str(image_path))
                    image_info.update(
                        {
                            "category": scene_category,
                            "original_image_path": str(image_path)
                        }
                    )
                
                    self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = self.category_space
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What category of scene is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "booth",
                "question": question,
                "wrong_choices_list": ["bedchamber", "beach", "bar"]
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
                