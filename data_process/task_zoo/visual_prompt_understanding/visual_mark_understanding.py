from pathlib import Path
import json

import mmcv

from base_dataset import BaseDataset


def encode_image_to_base64(img, target_size=-1):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    tmp = osp.join('/tmp', str(uuid4()) + '.jpg')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img.save(tmp)
    ret = encode_image_file_to_base64(tmp)
    os.remove(tmp)
    return ret


class vipbench(BaseDataset):
    DATA_METAINFO = {
        "bbox_image_path": "/path/to/lvlm_evaluation/data_process/data/ViP-Bench/bbox/images",
        "human_image_path": "/path/to/lvlm_evaluation/data_process/data/ViP-Bench/human/images",
        "anno_path": "/path/to/lvlm_evaluation/data/ViP-Bench/vip-bench-meta-data.json",
        "image_output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/image-to-image_translation/jigsaw_puzzle_solving/mscoco",
        "metainfo_output_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/image-to-image_translation/jigsaw_puzzle_solving",
        "sampling_num": 200,
        "dataset_description": "xxx",
        "visual_input_component": ["natural_image", "visual_mark"]
    }
    
    def parse_images_info(self):
        self.images_info = list()

        json_data_info = mmcv.load(self.anno_path)
        bbox_path = Path(self.anno_path).parent / "bbox"
        human_path = Path(self.anno_path).parent / "human"

        bbox_question_dict = dict()
        with open(bbox_path / "questions.jsonl", 'r') as file:
            for line in file:
                # 将每一行解析为 JSON 对象
                data = json.loads(line)
                bbox_question_dict[data["image"]] = data

        human_question_dict = dict()
        with open(human_path / "questions.jsonl", 'r') as file:
            for line in file:
                # 将每一行解析为 JSON 对象
                data = json.loads(line)
                human_question_dict[data["image"]] = data

        for _, image_info in json_data_info.items():
            pass
            image_source = image_info["image_source"]
            image = image_info["image"]
            question = image_info["question"]
            answer = image_info["answer"]

            # bbox
            pass
            bbox_ori_image_path = bbox_path / "images" / image
            bbox_ori_question = bbox_question_dict[image]['text']

            image_info = self.image_dict(str(bbox_ori_image_path))
            image_info.update(
                {
                    "original_image_path": str(bbox_ori_image_path),
                    "question": bbox_ori_question,
                    "answer": answer,
                    "filter_key":{
                    "type": "bbox",
                    "image_source": image_source
                }

                }
            )
            self.images_info.append(image_info)
        
            # human
            human_ori_image_path = human_path / "images" / image
            human_ori_question = human_question_dict[image]['text']

            image_info = self.image_dict(str(human_ori_image_path))
            image_info.update(
                {
                    "original_image_path": str(human_ori_image_path),
                    "question": human_ori_question,
                    "answer": answer,
                    "filter_key":{
                    "type": "human",
                    "image_source": image_source
                }

                }
            )
            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info):
        import json
        import mmcv
        from openai import OpenAI
        from prompt.utils import encode_image_to_base64

        question = image_info["question"]
        answer = image_info["answer"].split("<OR>")[0]
        prompt = f"Your task is to create options for a multi-choices question. I will provide an image, a question, and an answer. Please construct a few (at least one, but it can be 2 or 3) incorrect options based on the context. The wrong options should be misleading. \nQuestion: {question} \nAnswer: {answer}. Please be careful not to return the correct answer option, that is, the answer I provided you with. Please be aware that the number of incorrect options you generate should be based on the question and the answer. Typically, the number of incorrect options should be 1, 2, or 3."

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

        chatgpt_prompt = "Your task is to extract options from the provided text and return them in JSON format, encapsulated within a Python dictionary. This dictionary should contain only one key, options, whose corresponding value is a list, with each element of the list being an option. Please do not include option symbols such as A, B, C in the list elements."
        

        options = BaseDataset.openai_generate(chatgpt_prompt, response.choices[0].message.content)["options"]

        qa_json = {
            "question": question,
            "num_wrong_choices": len(options),
            "gt": answer,
            "wrong_choices_list": options
        }


        while True:
            try:
                qa_json = BaseDataset.post_process(qa_json)
                break
            except:
                pass

        return qa_json
    
