import random
from pathlib import Path

from base_dataset import BaseDataset

def check_duplicates(num, image_info, num_choices):
    wrong_choices_list = [random.sample(list(range(num)), len(image_info["mark_ids"])) for _ in range(num_choices - 1)]

    new_temp = []
    for choice in wrong_choices_list:
        if choice not in new_temp:
            new_temp.append(choice)
        else:
            return False
    
    return new_temp


class sombench_flickr30k_grounding(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/SoM-Bench/flickr30k_grounding/som_images",
        "sampling_num": 100,
        "visual_input_component": ["natural_image", "visual_mark"],
        "dataset_description": "xxx"
    }
    
    def parse_images_info(self):
        import mmcv
        self.images_info = list()

        for image_path in Path(self.image_path).iterdir():
            if image_path.suffix == ".json":
                continue
            if image_path.stem.endswith("box"):
                anno_path = image_path.with_suffix(".json").with_stem(image_path.stem[:-5])
            else:
                anno_path = image_path.with_suffix(".json")

            anno_info = mmcv.load(anno_path)
            if len(anno_info["gt_ids"]) < 3:
                continue

            image_info = self.image_dict(image_path)
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "question": anno_info['caption'],
                    "mark_ids": anno_info["gt_ids"]
                }
            )
            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, _):
        from prompt.utils import encode_image_to_base64
        from prompt.utils import generate_incorrect_bounding_boxes
        from openai import OpenAI
        num_choices = 4
        width, height = image_info["width"], image_info["height"]
        question = f"I have labeled a bright numeric ID at the center for each visual object in the image. And I will provide you with a sentence that contains several objects enclosed in parentheses. Please identify the corresponding region IDs for these objects in the image in the order they appear. Sentence: {image_info['question']}"

        prompt = f"Please tell me how many highlighted objects in this image. Your answer is just a number without any other output."

        # Path to your image
        image_path = image_info["original_image_path"]

        from PIL import Image
        # Getting the base64 string
        base64_image = encode_image_to_base64(Image.open(image_path), 768)

        openai_client = OpenAI()
        response = openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "image_detail": "low"
                        }
                    },
                ],
                }
            ],
            max_tokens=300,
        )

        while True:
            try:
                import random
                num = int(response.choices[0].message.content)
                num = max(max(image_info['mark_ids']), num)
                # num = 1 if num == 0 else num
                if num == 1:
                    return None
                wrong_choices_list = check_duplicates(num, image_info, num_choices)
                assert wrong_choices_list
                # wrong_choices_list = 
                # wrong_choices_list = generate_incorrect_bounding_boxes(new_gt, width, height, num_choices-1)
                qa_json = {
                    "question": question,
                    "num_wrong_choices": num_choices - 1,
                    "gt": image_info["mark_ids"], 
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                pass

        return qa_json
    
    @staticmethod
    def modify_qa(qa_json):
        qa_json["question"] = f"I have labeled a bright numeric ID at the center for each visual object in the image. And I will provide you with a sentence that contains several objects enclosed in parentheses. Please identify the corresponding region IDs for these objects in the image in the order they appear. Sentence: {qa_json['question']}"

        return qa_json

class sombench_refcocog_refseg(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/SoM-Bench/refcocog_refseg/som_images",
        "sampling_num": 100,
        "visual_input_component": ["natural_image", "visual_mark"],
        "dataset_description": "xxx"
    }
    
    def parse_images_info(self):
        import mmcv
        self.images_info = list()

        for image_path in Path(self.image_path).iterdir():
            if image_path.suffix == ".json":
                continue
            if image_path.stem.endswith("box"):
                anno_path = image_path.with_suffix(".json").with_stem(image_path.stem[:-5])
            else:
                anno_path = image_path.with_suffix(".json")

            anno_info = mmcv.load(anno_path)

            for anno_ in anno_info:
                image_info = self.image_dict(image_path)
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "question": anno_['text'],
                        "mark_ids": anno_["ref_id"]
                    }
                )
                self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, _):
        from prompt.utils import encode_image_to_base64
        from prompt.utils import generate_incorrect_bounding_boxes
        from openai import OpenAI
        num_choices = 4
        import random
        width, height = image_info["width"], image_info["height"]
        question = f"I have labeled a bright numeric ID at the center for each visual object in the image. And I will provide you with a sentence that describe a specific object in this image. Please identify the corresponding region ID for this objects in the image. Sentence: {random.choice(image_info['question'])}"

        prompt = f"Please tell me how many highlighted objects in this image. Your answer is just a number without any other output."

        # Path to your image
        image_path = image_info["original_image_path"]

        from PIL import Image
        # Getting the base64 string
        base64_image = encode_image_to_base64(Image.open(image_path), 768)

        openai_client = OpenAI()
        response = openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "image_detail": "low"
                        }
                    },
                ],
                }
            ],
            max_tokens=300,
        )

        while True:
            try:
                import random
                num = int(response.choices[0].message.content)
                num = max(image_info['mark_ids'], num)
                # num = 1 if num == 0 else num
                if num == 1:
                    return None
                elif num == 2:
                    num_choices = 2
                elif num == 3:
                    num_choices = 3
                
                wrong_choices_list = [random.choice(list(range(num))) for _ in range(num_choices - 1)]
                new_temp = []
                for choice in wrong_choices_list:
                    if choice not in new_temp:
                        new_temp.append(choice)
                    else:
                        wrong_choices_list = None
                    
                assert wrong_choices_list
                # wrong_choices_list = 
                # wrong_choices_list = generate_incorrect_bounding_boxes(new_gt, width, height, num_choices-1)
                qa_json = {
                    "question": question,
                    "num_wrong_choices": num_choices - 1,
                    "gt": image_info["mark_ids"], 
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                pass

        return qa_json
    
    @staticmethod
    def modify_qa(qa_json):
        qa_json["question"] = f"I have labeled a bright numeric ID at the center for each visual object in the image. And I will provide you with a sentence that contains several objects enclosed in parentheses. Please identify the corresponding region IDs for these objects in the image in the order they appear. Sentence: {random.choice(qa_json['question'])}"

        return qa_json
