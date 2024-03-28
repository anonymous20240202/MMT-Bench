from pathlib import Path

import mmcv

from base_dataset import BaseDataset


class hicodet_hiu(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/lvlm_evaluation/data/hico_20160224_det/annotations/test_hico.json",
        "image_path": "/path/to/lvlm_evaluation/data/hico_20160224_det/images/test2015",
        "image_save_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/relation_reasoning/human-object_interaction_recognition",
        "sampling_num": 57,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.category_space = [
            'adjust', 'assemble', 'block', 'blow', 'board', 'break',
            'brush_with', 'buy', 'carry', 'catch', 'chase', 'check',
            'clean', 'control', 'cook', 'cut', 'cut_with', 'direct',
            'drag', 'dribble', 'drink_with', 'drive', 'dry', 'eat', 'eat_at',
            'exit', 'feed', 'fill', 'flip', 'flush', 'fly', 'greet', 'grind',
            'groom', 'herd', 'hit', 'hold', 'hop_on', 'hose',
            'hug', 'hunt', 'inspect', 'install', 'jump', 'kick', 'kiss',
            'lasso', 'launch', 'lick', 'lie_on', 'lift', 'light',
            'load', 'lose', 'make', 'milk', 'move', 'no_interaction',
            'open', 'operate', 'pack', 'paint', 'park', 'pay',
            'peel', 'pet', 'pick', 'pick_up', 'point', 'pour', 'pull', 'push',
            'race', 'read', 'release', 'repair', 'ride',
            'row', 'run', 'sail', 'scratch', 'serve', 'set', 'shear', 'sign',
            'sip', 'sit_at', 'sit_on', 'slide', 'smell', 'spin', 'squeeze',
            'stab', 'stand_on', 'stand_under', 'stick', 'stir', 'stop_at', 'straddle', 'swing', 'tag',
            'talk_on', 'teach', 'text_on', 'throw', 'tie', 'toast', 'train', 'turn', 'type_on', 'walk', 'wash',
            'watch', 'wave', 'wear', 'wield', 'zip'
        ]
        self.dataset_info["category_space"] = [
            'hug', 'greet', 'kiss', 'shake hanks', 'tag', 'talk', 'text on', 'teach', 'stab', 'hold', 'lick'
        ]
    
    def parse_images_info(self):
        self.images_info = list()
        json_data_list = mmcv.load(self.anno_path)

        for image_info in json_data_list:
            ori_image_path = Path(self.image_path) / image_info["file_name"]

            if len(image_info["hoi_annotation"]) != 1:
                continue

            subject_id = image_info["hoi_annotation"][0]["subject_id"]
            object_id = image_info["hoi_annotation"][0]["object_id"]

            subject_bbox = image_info["annotations"][subject_id]['bbox']
            # subject_category = 'person'
            if image_info["annotations"][object_id]['category_id'] != 1:
                # 只考虑人与人的交互
                continue
            object_bbox = image_info["annotations"][object_id]['bbox']

            category_id = image_info["hoi_annotation"][0]["category_id"]
            category = self.category_space[category_id - 1]

            print(category)

            # _image_info = copy.deepcopy(self.image_dict)
            image_dict = self.image_dict("")
            image_dict.update({
                "original_image_path": str(ori_image_path),
                "category": category,
                "bounding_box_coordinates": {
                "subject": subject_bbox,
                "object": object_bbox}
            })
            self.images_info.append(image_dict)

    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import mmcv
        import numpy as np
        import json
        num_choices = 4
        question = "What is the interaction relationship between these two people (marked as RED bounding boxes)?"

        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "greet",
                "question": question,
                "wrong_choices_list": ["hug", "tag", "teach"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["category"],
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)


        BaseDataset.exist_or_mkdir(save_image_path)
        bbox_list = [np.array([image_info['bounding_box_coordinates']['subject']]), np.array([image_info['bounding_box_coordinates']['object']])]
        color_list = ["red", "red"]
        # width, height = image_info["width"], image_info["height"]
        # for box in image_info["bounding_box_coordinates"]:
        #     bbox_list.append([box[0] * width, box[1] * height, box[2] * width, box[3] * height])
        new_image_path = str(Path(save_image_path) / BaseDataset.new_image_name())
        mmcv.imshow_bboxes(image_info["original_image_path"], bbox_list, show=False, out_file=new_image_path, thickness=2, colors=color_list)

        i = 0
        while i <= 10:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json, question=question)
                qa_json["original_image_path"] = new_image_path
                break
            except:
                i += 1

        return qa_json


class bit(BaseDataset):
    DATA_METAINFO = {
            "anno_path": "/path/to/lvlm_evaluation/data/BIT/BIT-anno/tidy_anno",
            "image_path": "/path/to/lvlm_evaluation/data/BIT/Bit-frames",
            "image_save_path": "/path/to/lvlm_evaluation/taskonomy_evaluation_data/relation_reasoning/human_interaction_recognition",
            "sampling_num": 143,
            "visual_input_component": ["natural_image"],
            "dataset_description": "xxx"}
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.category_space = ["bend", "box", "handshake", "highfive", "hug", "kick", "pat"]
        
        self.dataset_info["category_space"] = self.category_space
    
    def parse_images_info(self):
        self.images_info = []

        for action_path in Path(self.anno_path).iterdir():
            # for video_path
            action_name = action_path.name
            for video_anno in action_path.iterdir():
                video_name = video_anno.stem

                anno_list = mmcv.list_from_file(video_anno)

                for anno_line in anno_list:
                    if anno_line.startswith("frame"):
                        num_bbox = int(anno_line.split(' ')[-1])
                        frame_id = int(anno_line.split(' ')[1])

                        width, height = self.get_image_width_height(Path(self.image_path) / action_name / video_name / f'{str(frame_id).zfill(4)}.jpg')
                        category_list = []
                        bbox_list = []
                        continue
                    
                    if num_bbox > 0:
                        human_id, x1, y1, x2, y2, action_category = anno_line.split()

                        category_list.append(action_category)
                        bbox_list.append([float(x1), float(y1), float(x2), float(y2)])

                        num_bbox -= 1
                    
                    if num_bbox == 0:
                        # 最后一个bbox已经遍历完了
                        not_c_count = sum(1 for i, element in enumerate(category_list) if element != "no_action")
                        if not_c_count == 0:
                            continue
                        not_c_indices = [i for i, element in enumerate(category_list) if element != "no_action"]
                        category = False
                        for i in not_c_indices:
                            if category_list[i] in self.category_space:
                                category = category_list[i]
                                break
                        if not category:
                            continue
                        
                        image_info = self.image_dict("")
                        image_info.update(
                            {
                                "original_image_path": str(Path(self.image_path) / action_name / video_name / f'{str(frame_id).zfill(4)}.jpg'),
                                "category": category,
                                "bounding_box_coordinates": [
                                    bbox_list[i] for i in not_c_indices
                                ]
                            }
                        )
                        self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import mmcv
        import numpy as np
        import json
        num_choices = 4
        question = "What is the interaction relationship between these two people (marked as RED bounding boxes)?"

        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "bend",
                "question": question,
                "wrong_choices_list": ["box", "handshake", "highfive"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["category"],
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)


        BaseDataset.exist_or_mkdir(save_image_path)
        bbox_list = image_info["bounding_box_coordinates"]
        new_image_path = str(Path(save_image_path) / BaseDataset.new_image_name())
        mmcv.imshow_bboxes(image_info["original_image_path"], np.array(bbox_list), show=False, out_file=new_image_path, thickness=2, colors="red")

        i = 0
        while i <= 10:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json, question=question)
                qa_json["origianl_image_path"] = new_image_path
                break
            except:
                i += 1

        return qa_json
