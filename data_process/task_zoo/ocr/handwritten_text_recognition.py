from pathlib import Path

import mmcv

from base_dataset import BaseDataset


class iam_line(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/ocr/data/HTR/IAM/images_line",
        "anno_path": "/path/to/lvlm_evaluation/data/ocr/data/HTR/IAM/GT_line",
        "sampling_num": 100,
        "visual_input_component": ["text-rich_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        for image_path in Path(self.image_path).iterdir():
            original_image_path = str(image_path)
            
            text_path = Path(self.anno_path) / f"{image_path.stem}.txt"
            pass

            text_list = mmcv.list_from_file(text_path)
            assert len(text_list) == 1
            text = text_list[0]

            image_info = self.image_dict(original_image_path)
            image_info.update(
                {
                    "original_image_path": original_image_path,
                    "text": text
                }
            )
            self.images_info.append(image_info)
        
    @staticmethod
    def generate_qa(image_info, dataset_info, _):
        import json
        import mmcv

        num_choices = 4
        question = "Recognize the text in the image."
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/handwritten_text_recognition.txt'))

        input_json = {
            "question": question,
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "him?\" John didn't answer. There was no point",
                "question": question,
                "wrong_choices_list": [
                    "him?\" John didn't reply. There was no purpose.", 
                    "him?\" John did not respond. There was no sense.", 
                    "he?\" John didn't reply. There was no reason."
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


class iam_page(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/ocr/data/HTR/IAM/images_page",
        "anno_path": "/path/to/lvlm_evaluation/data/ocr/data/HTR/IAM/GT_page",
        "sampling_num": 100,
        "visual_input_component": ["text-rich_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        for image_path in Path(self.image_path).iterdir():
            original_image_path = str(image_path)
            
            text_path = Path(self.anno_path) / f"{image_path.stem}.txt"

            text_list = mmcv.list_from_file(text_path)

            text = "\n".join(text_list)
            image_info = self.image_dict(original_image_path)
            image_info.update(
                {
                    "original_image_path": original_image_path,
                    "text": text
                }
            )
            self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, _):
        import json
        import mmcv

        num_choices = 4
        question = "Recognize the text in the image. To create a new line, use \"\n\"."
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/handwritten_text_recognition.txt'))

        input_json = {
            "question": question,
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "But since starting salaries would depend on grade A\nor B in the finals next May, and since mating\nprospects would depend upon salaries, scholarship for\nthese fine young people was closely geared to\neconomic and biological ends which, essentially,\nwere really means. So, seeing them revolve in\ncircles, Harry had the feeling that Moke (or what\nMoke consciously or unconsciously symbolised, any-\nway in Harry's mind) had these splendid young\npeople by the short hairs, and was diverting them ...",
                "question": question,
                "wrong_choices_list": [
                    "But since starting salaries would depend on grade A\nor B in the finals next May, and since mating\nprospects would depend on salaries, scholarship for\nthese fine young people was closely geared to\neconomic and biological ends which, essentially,\nwere really means. So, seeing them revolve in\ncircles, Harry had the feeling that Moke (or what\nMoke consciously or unconsciously symbolised, any-\nway in Harry's mind) had these splendid young\npeople by the short hairs, and was diverting them ...",
                    "But since starting salaries would depend on grade A\nor B in the finals next May, and since mating\nprospects would depend upon salaries, scholarship for\nthese fine young people was closely geared to\neconomic and biological ends which, essentially,\nwere really means. So, seeing them revolve in\ncircles, Harry had the feeling that Moke (or what\nMoke consciously or unconsciously symbolised, any-\nway in Harry's mind) had these splendid young\npeople by the short hair, and was diverting them ...",
                    "But since starting salaries would depend on grade A\nor B in the finals next May, and since mating\nprospects would depend upon salaries, scholarship for\nthese fine young people was closely geared to\neconomic and biological ends which, essentially,\nwere real means. So, seeing them revolve in\ncircles, Harry had the feeling that Moke (or what\nMoke consciously or unconsciously symbolised, any-\nway in Harry's mind) had these splendid young\npeople by the short hairs, and was diverting them ..."
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
    