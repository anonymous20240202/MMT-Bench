from pathlib import Path

from base_dataset import BaseDataset


def remove_trailing_digits(input_str):
    # 从字符串末尾开始遍历，找到第一个非数字字符的索引
    for i in range(len(input_str) - 1, -1, -1):
        if not input_str[i].isdigit():
            return input_str[:i + 1]  # 返回去除数字部分的子字符串
    return input_str  # 如果字符串没有非数字字符，则返回原始字符串


class kth(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/attribute_recognition/texture/valid",
        "sampling_num": 50,
        "visual_input_component": ["natural_image", "grayscale_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for category_path in Path(self.image_path).iterdir():
            if remove_trailing_digits(category_path.name.split('_')[0]) != "KTH":
                continue
            if len(category_path.name.split('_')) == 3:
                category_name = " ".join(category_path.name.split('_')[-2:])
            else:
                category_name = category_path.name.split('_')[1]
            dataset_name = category_path.name.split('_')[0]
            if remove_trailing_digits(dataset_name) != "KTH":
                continue
            category_name = remove_trailing_digits(category_name)
            self.category_space.append(category_name)
            
            for image_path in category_path.iterdir():
                image_info = self.image_dict(str(image_path))
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "category": category_name
                    }
                )
                self.images_info.append(image_info)

            self.dataset_info["category_space"] = list(set(self.category_space))

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the material of the texture in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "orange peel",
                "question": question,
                "wrong_choices_list": ["linen", "cotton", "sponge"]
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


class kyberge(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/attribute_recognition/texture/valid",
        "sampling_num": 50,
        "visual_input_component": ["natural_image", "grayscale_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for category_path in Path(self.image_path).iterdir():
            if remove_trailing_digits(category_path.name.split('_')[0]) != "Kyberge":
                continue
            dataset_name, category_name = category_path.name.split('_')
            category_name = remove_trailing_digits(category_name)
            self.category_space.append(category_name)
            
            for image_path in category_path.iterdir():
                image_info = self.image_dict(str(image_path))
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "category": category_name
                    }
                )
                self.images_info.append(image_info)

            self.dataset_info["category_space"] = list(set(self.category_space))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the material of the texture in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "wall",
                "question": question,
                "wrong_choices_list": ["grass", "stone", "sand"]
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


class uiuc(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/attribute_recognition/texture/valid",
        "sampling_num": 50,
        "visual_input_component": ["natural_image", "grayscale_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for category_path in Path(self.image_path).iterdir():
            if remove_trailing_digits(category_path.name.split('_')[0]) != "UIUC":
                continue
            dataset_name, category_name = category_path.name.split('_')
            category_name = remove_trailing_digits(category_name)
            self.category_space.append(category_name)
            
            for image_path in category_path.iterdir():
                image_info = self.image_dict(str(image_path))
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "category": category_name
                    }
                )
                self.images_info.append(image_info)

            self.dataset_info["category_space"] = list(set(self.category_space))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the material of the texture in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "granite",
                "question": question,
                "wrong_choices_list": ["upholstery", "pebbles", "carpet"]
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


class opensurfaces(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/opensurfaces",
        "sampling_num": 50,
        "visual_input_component": ["natural_image", "visual_mark"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for category_path in Path(self.image_path).iterdir():
            category_name = category_path.name
            self.category_space.append(category_name)

            for image_path in category_path.iterdir():
                image_info = self.image_dict(str(image_path))
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "category": category_name
                    }
                )
                self.images_info.append(image_info)

            self.dataset_info["category_space"] = list(set(self.category_space))
    

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What is the material of the highlighted object (red edge) in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "leather",
                "question": question,
                "wrong_choices_list": ["fur", "brick", "glass"]
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
