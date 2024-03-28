from pathlib import Path

import mmcv

from base_dataset import BaseDataset


class sketch2code_kaggle(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/visual_code/data",
        "sampling_num": 200,
        "visual_input_component": ["abstract_image", ],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        for image_path in Path(self.image_path).iterdir():
            if image_path.suffix == '.png':
                anno_gui_path = image_path.with_suffix('.gui')

                gui_text_list = mmcv.list_from_file(anno_gui_path)
                gui_text = '\n'.join(gui_text_list)

                image_info = self.image_dict("")
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "text": gui_text
                    }
                )
                self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import json
        import mmcv

        num_choices = 4
        question = "Convert this sketch GUI to layout code."
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/screenshot2code.txt'))

        input_json = {
            "question": question,
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "stack {\nrow {\nswitch\n}\nrow {\nlabel, btn\n}\n}\nfooter {\nbtn-dashboard, btn-notifications, btn-home\n}",
                "question": question,
                "wrong_choices_list": [
                        "stack {\nrow {\nswitch\n}\nrow {\nlabel, btn\n}\n}\nfooter {\nbtn-dashboard, btn-notifications, btn-home, btn-settings\n}",
                        "stack {\nrow {\nswitch\n}\nrow {\nlabel, btn\n}\n}\nfooter {\nbtn-dashboard, btn-home\n}",
                        "stack {\nrow {\nswitch\n}\nrow {\nlabel, btn\n}\n}\nfooter {\nbtn-dashboard, btn-notifications, btn-home\n}\nbtn-settings"
                    ]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["text"],
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
