from pathlib import Path

import numpy as np

from base_dataset import BaseDataset
from tqdm import tqdm


class wikiart(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/lvlm_evaluation/data/wikiart/wikiart/data/train-00000-of-00072.parquet",
        "anno_file_path": "/path/to/lvlm_evaluation/data/wikiart/wikiart/dataset_infos.json",
        "save_image_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/painting_recognition/wikiart",
        "sampling_num": 80,
        "visual_input_component": ["painting_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import pandas as pd
        import mmcv
        self.images_info = list()

        df = pd.read_parquet(self.anno_path)
        anno_file = mmcv.load(self.anno_file_path)

        artist_list = anno_file["huggan--wikiart"]["features"]["artist"]
        genre_list =  anno_file["huggan--wikiart"]["features"]["genre"]
        style_list =  anno_file["huggan--wikiart"]["features"]["style"]

        self.images_info = list()

        for _, row in enumerate(tqdm(df.itertuples())):

            image_rgb = mmcv.imfrombytes(row.image["bytes"])
            original_image_path = Path(self.save_image_path) / self.new_image_name()
            self.save_rgb_image(image_rgb, original_image_path)
            artist = artist_list['names'][row.artist]
            genre = genre_list['names'][row.genre]
            style = style_list['names'][row.style]

            image_info = self.image_dict(str(original_image_path))
            image_info.update(
                {
                    "original_image_path": str(original_image_path),
                    "artist": artist,
                    "genre": genre,
                    "style": style
                }
            )

            self.images_info.append(image_info)
        
        self.dataset_info["artist_list"] = artist_list["names"]
        self.dataset_info["genre_list"] = genre_list["names"]
        self.dataset_info["style_list"] = style_list["names"]
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv
        import random
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        num_choices = 4
        name = random.choice(['genre', 'style'])
        if name == "genre":
            question = "What is the genre of the painting in the picture?"
            input_json = {
                "question": question,
                "category_space": dataset_info["genre_list"],
                "example_dict": {
                    "num_wrong_choices": num_choices - 1,
                    "gt": "illustration",
                    "question": question,
                    "wrong_choices_list": ["portrait", "religious_painting", "cityscape"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["genre"],
                "question": question,
            }
            }
        else:
            question = "What is the style of the painting in the picture?"
            input_json = {
                "question": question,
                "category_space": dataset_info["style_list"],
                "example_dict": {
                    "num_wrong_choices": num_choices - 1,
                    "gt": "Analytical_Cubism",
                    "question": question,
                    "wrong_choices_list": ["Abstract_Expressionism", "Baroque", "Cubism"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["style"],
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


class best_artwork_of_all_time(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/painting/best_artwork_of_all_time/images/images",
        "sampling_num": 50,
        "visual_input_component": ["painting_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        self.category_lsit = []
        for artist_path in Path(self.image_path).iterdir():
            artist_name = " ".join(artist_path.stem.split("_"))
            self.category_lsit.append(artist_name)
            for image_path in artist_path.iterdir():

                image_info = self.image_dict(str(image_path))
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "catgeory": artist_name
                    }
                )
                self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = self.category_lsit
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the artist of the painting in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Paul Gauguin",
                "question": question,
                "wrong_choices_list": ["Edgar Degas", "Titian", "Diego Velazquez"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["catgeory"],
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


class van_gogh_paintings_dataset(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/painting/van_gogh_paintings_dataset",
        "sampling_num": 50,
        "visual_input_component": ["painting_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import re
        self.images_info = list()
        self.category_space = []

        for painting_path in Path(self.image_path).iterdir():
            for image_path in painting_path.iterdir():
                if bool(re.search(r'\d$', image_path.stem)):
                    continue

                image_info = self.image_dict(str(image_path))
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "category": image_path.stem
                    }
                )
                self.images_info.append(image_info)
                self.category_space.append(image_path.stem)
        
        self.dataset_info["category_space"] = self.category_space
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the name of the painting in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Portrait of Patience Escalier, Shepherd in Provence",
                "question": question,
                "wrong_choices_list": ["The Poet s Garden", "The Brothel", "A L Arlesienne Madame Ginoux with Gloves and Umbre"]
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


class chinese_patinting_internet(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/painting/chinese_painting_internet/",
        "sampling_num": 50,
        "visual_input_component": ["painting_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import re
        self.images_info = list()
        self.category_space = []

        for painting_path in Path(self.image_path).iterdir():
            image_info = self.image_dict(str(painting_path))
            image_info.update(
                {
                    "original_image_path": str(painting_path),
                    "category": painting_path.stem
                }
            )
            self.images_info.append(image_info)
            self.category_space.append(painting_path.stem)
        
        self.dataset_info["category_space"] = self.category_space

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the name of the painting in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "Early Spring",
                "question": question,
                "wrong_choices_list": ["The Sound of Autumn", "Listening to the Wind in Pines", "Xiao and Xiang Rivers"]
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
