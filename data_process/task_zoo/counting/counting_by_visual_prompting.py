from pathlib import Path

import mmcv
import numpy as np

from base_dataset import BaseDataset


class fsc147(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/counting/images_384_VarV2",
        "anno_path": "/path/to/lvlm_evaluation/data/counting/annotation_FSC147_384.json",
        "density_map": "/path/to/lvlm_evaluation/data/counting/gt_density_map_adaptive_384_VarV2",
        "sampling_num": 200,
        "visual_input_component": ["natural_image", "visual_mark"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        anno_info_lsit = mmcv.load(self.anno_path)
        for image_name, image_info_dict in anno_info_lsit.items():
            box_examples = image_info_dict["box_examples_coordinates"]

            new_box_examples = []
            for box in box_examples:
                temp_list = []
                temp_list.extend(box[0])
                temp_list.extend(box[2])
                new_box_examples.append(temp_list)

            counting_num = int(np.load((Path(self.density_map) / image_name).with_suffix('.npy')).sum())

            if counting_num >= 30:
                continue

            image_info = self.image_dict(str(Path(self.image_path) / image_name))
            image_info.update(
                {
                    "box_examples": new_box_examples,
                    "counting_num": counting_num,
                    "original_image_path": str(Path(self.image_path) / image_name)
                }
            )
            self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        def get_answer(gt, rangee=4, num=3):
            import random
            a = list(range(max(0, gt - rangee), gt + rangee))
            a.remove(gt)
            return random.sample(a, k=num)

        question = f"Please state the number of objects of the category marked by the green bounding box in the image."

        import mmcv
        import numpy as np
        BaseDataset.exist_or_mkdir(save_image_path)
        mmcv.imshow_bboxes(image_info["original_image_path"], np.array(image_info['box_examples']), show=False, out_file=Path(save_image_path) / BaseDataset.new_image_name())

        qa_json = {
            "question": question,
            "num_wrong_choices": 3,
            "wrong_choices_list": get_answer(image_info["counting_num"]),
            "gt": image_info["counting_num"]
        }

        while True:
            try:
                qa_json = BaseDataset.post_process(qa_json)
                break
            except:
                pass

        return qa_json
