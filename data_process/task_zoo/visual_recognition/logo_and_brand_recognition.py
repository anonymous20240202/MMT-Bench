from pathlib import Path

from base_dataset import BaseDataset


class fake_real_logo_detection_dataset(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/logo/fake_and_true/output",
        "sampling_num": 100,
        "visual_input_component": ["synthetic_image", "low-resolution"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for logo_path in Path(self.image_path).iterdir():
            logo_category = logo_path.name
            self.category_space.append(logo_category)

            for image_path in logo_path.iterdir():
                image_info = self.image_dict(image_path)
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "category": logo_category
                    }
                )

                self.images_info.append(image_info)

        self.dataset_info["category_space"] = self.category_space

    @staticmethod
    def generate_qa(image_info, dataset_info, _):
        import json
        import mmcv

        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))
        question = "What brand is shown in the logo depicted in the picture?"
        num_choices = 4

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Google",
                "question": question,
                "wrong_choices_list": ["Facebook", "Tiktok", "Meta"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["category"],
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


class flickr_sport_logos_10(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/logo/FlickrSportLogos-10/dataset/JPEGImages",
        "anno_path": "/path/to/lvlm_evaluation/data/logo/FlickrSportLogos-10/dataset/Annotations",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import xml.etree.ElementTree as ET
        self.images_info = list()
        self.category_space = []

        for image_path in Path(self.image_path).iterdir():
            pass
            
            anno_path = Path(self.anno_path) / f"{image_path.stem}.xml"

            tree = ET.parse(anno_path)
            root = tree.getroot()

            objects = root.findall('object')
            if len(objects) != 1:
                continue
            for object in objects:
                object_name = object.find('name').text
            
            image_info = self.image_dict(image_path)
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "category": object_name
                }
            )

            self.category_space.append(object_name)

            self.images_info.append(image_info)

        self.dataset_info["category_space"] = list(set(self.category_space))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, _):
        import json
        import mmcv

        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))
        question = "What brand is shown in the logo depicted in the picture?"
        num_choices = 4

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "puma",
                "question": question,
                "wrong_choices_list": ["nike", "erke", "adidas"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["category"],
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


class car_logos_dataset(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/logo/Car-Logos-Dataset/optimized",
        "sampling_num": 50,
        "visual_input_component": ["synthetic"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for image_path in Path(self.image_path).iterdir():
            category = image_path.stem

            image_info = self.image_dict(image_path)
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "category": category
                }
            )

            self.category_space.append(category)

            self.images_info.append(image_info)

        self.dataset_info["category_space"] = list(set(self.category_space))

    @staticmethod
    def generate_qa(image_info, dataset_info, _):
        import json
        from prompt.base_system_prompt import single_choice_classification_sys_prompt

        sys_prompt = single_choice_classification_sys_prompt
        input_json = {
            "question": "What brand is shown in the logo depicted in the picture?",
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_choices": 4,
                "gt_category": "byd",
                "question": "What brand is shown in the logo depicted in the picture?",
                "choice_a": "tesla",
                "choice_b": "byd",
                "choice_c": "bmw",
                "choice_d": "benz",
                "gt_choice": "b"
            },
            "query_dict": {
                "num_choices": 4,
                "gt_category": image_info["category"],
                "question": "What brand is shown in the logo depicted in the picture?",
            }
        }

        user_prompt = json.dumps(input_json)

        qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
        return qa_json

class logo627(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/logo/logo-627/logo-images-dataset-master",
        "sampling_num": 50,
        "visual_input_component": ["synthetic"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for PPath in Path(self.image_path).iterdir():

            for logo_path in PPath.iterdir():
                # if logo_path.suffix != ".png":
                #     continue
                pass
                category = logo_path.name
                self.category_space.append(category)

                for image_path in logo_path.iterdir():
                    if image_path.suffix != ".png":
                        continue
                    image_info = self.image_dict(image_path)
                    image_info.update(
                        {
                            "original_image_path": str(image_path),
                            "category": category
                        }
                    )

                    self.images_info.append(image_info)

        self.dataset_info["category_space"] = list(set(self.category_space))
    
    @staticmethod
    def generate_qa(image_info, dataset_info):
        import json
        from prompt.base_system_prompt import single_choice_classification_sys_prompt

        sys_prompt = single_choice_classification_sys_prompt
        input_json = {
            "question": "What brand is shown in the logo depicted in the picture?",
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_choices": 4,
                "gt_category": "byd",
                "question": "What brand is shown in the logo depicted in the picture?",
                "choice_a": "tesla",
                "choice_b": "byd",
                "choice_c": "bmw",
                "choice_d": "benz",
                "gt_choice": "b"
            },
            "query_dict": {
                "num_choices": 4,
                "gt_category": image_info["category"],
                "question": "What brand is shown in the logo depicted in the picture?",
            }
        }

        user_prompt = json.dumps(input_json)

        qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
        return qa_json