from pathlib import Path

import mmcv

from base_dataset import BaseDataset


class social_relation_dataset(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/lvlm_evaluation/data/OpenDataLab___Social_Relation/raw/interpersonal_relation_dataset/testing.txt",
        "image_path": "/path/to/lvlm_evaluation/data/OpenDataLab___Social_Relation/raw/interpersonal_relation_dataset/img",
        "save_image_path": "/path/to/lvlm_evaluation/data_process/taskonomy_evaluation_data/relation_reasoning/social_relation_recognition/images",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        a = mmcv.load("/path/to/data/markllava/finetune/visual_genome/train.json")
        super().parse_dataset_info()
        self.category_space = [
            'dominated', 'assured', 'demonstrative', 'involved', 'friendly', 'warm', 'trusting', 'competitive'
        ]
        self.dataset_info["category_space"] = self.category_space
        self.dataset_info["category_num"] = "multiple"
    
    def parse_images_info(self):
        self.images_info = list()
        anno_data_list = mmcv.list_from_file(self.anno_path)
        # json_data_list = mmcv.load(self.anno_path)
        for anno_data in anno_data_list:
            anno_list = anno_data.strip().split()
            original_image_path = Path(self.image_path) / anno_list[0]
            width, height = self.get_image_width_height(original_image_path)

            x, y, w, h = [float(i) for i in anno_list[1:5]]
            face_1_box = [x / width, y / height, (x + w) / width, (y + h) / height]

            x, y, w, h = [int(i) for i in anno_list[5:9]]
            face_2_box = [x / width, y / height, (x + w) / width, (y + h) / height]

            label_list = anno_list[9: ]
            label_list = [int(i) for i in label_list]

            category_list = [self.category_space[i] for i, flag in enumerate(label_list) if flag == 1]

            image_info = self.image_dict("")
            image_info.update(
                {
                    "original_image_path": str(original_image_path),
                    "bounding_box_coordinates": [face_1_box, face_2_box],
                    "category_list": category_list
                }
            )

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import mmcv
        import numpy as np
        import json
        num_choices = 4
        question = "What is the social relationship between these two people (with each person's face marked with a GREEN bounding box)?"

        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/multi_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "dominated,demonstrative,friendly,involved,competitive",
                "question": question,
                "wrong_choices_list": ["assured,demonstrative,involved,friendly,warm,trusting", "assured", "friendly,trusting,demonstrative"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": ",".join(image_info["category_list"]),
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)


        BaseDataset.exist_or_mkdir(save_image_path)
        bbox_list = []
        width, height = image_info["width"], image_info["height"]
        for box in image_info["bounding_box_coordinates"]:
            bbox_list.append([box[0] * width, box[1] * height, box[2] * width, box[3] * height])
        
        new_image_path = str(Path(save_image_path) / BaseDataset.new_image_name())
        mmcv.imshow_bboxes(image_info["original_image_path"], np.array(bbox_list), show=False, out_file=new_image_path, thickness=2)

        i = 0
        while i <= 10:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json['gt'] = ",".join(image_info["category_list"])

                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1
        qa_json["original_image_path"] = new_image_path

        return qa_json
