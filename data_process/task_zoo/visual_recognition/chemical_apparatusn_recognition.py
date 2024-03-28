from pathlib import Path
from collections import OrderedDict

from base_dataset import BaseDataset


class chemical_apparatus_image_dataset(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/chemical/Chemical_Apparatus_Image_Dataset/Test/Images",
        "anno_path": "/path/to/lvlm_evaluation/data_process/data/chemical/Chemical_Apparatus_Image_Dataset/Test/Labels",
        "sampling_num": 200,
        "url": "",
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.category_space = [
            "conical beaker",
            "erlenmeyer flask",
            "reagent bottle",
            "pipette",
            "eggplant shaped flask",
            "separatory funnel"
        ]
        self.dataset_info["category_space"] = self.category_space
    
    def parse_images_info(self):
        import mmcv

        from PIL import Image

        def crop_and_save_image_center(image_path, coordinates, save_path):
            """
            Crops an image based on given center coordinates and dimensions (cxcywh) and saves the cropped image to a specified path.
            The coordinates are in the format (cx, cy, w, h) where each value is a floating-point number between 0 and 1.

            :param image_path: Path to the original image.
            :param coordinates: A tuple (cx, cy, w, h) of the crop box center and size.
            :param save_path: Path where the cropped image will be saved.
            """
            # Open the image
            with Image.open(image_path) as img:
                width, height = img.size

                # Convert fractional coordinates to absolute pixel coordinates
                cx, cy, w, h = coordinates
                x = int((cx - w / 2) * width)
                y = int((cy - h / 2) * height)
                w = int(w * width)
                h = int(h * height)

                # Define the crop box
                crop_box = (x, y, x + w, y + h)

                # Crop and save the image
                cropped_img = img.crop(crop_box)
                cropped_img.save(save_path)

        self.images_info = list()
        self.category_space = []
        for image_path in Path(self.image_path).iterdir():
            label_path = (Path(self.anno_path) / image_path.name).with_suffix(".txt")

            label_list = mmcv.list_from_file(label_path)

            for label_line in label_list:
                category, x, y, w, h = label_line.split()

                category = int(category)
                x = float(x)
                y = float(y)
                w = float(w)
                h = float(h)

                if category == 0:
                    continue

                new_image_name = self.new_image_name()
                crop_and_save_image_center(image_path, (x, y, w, h), f"/path/to/lvlm_evaluation/data_process/data/chemical/crop_images/{new_image_name}")

                image_info = self.image_dict(f"/path/to/lvlm_evaluation/data_process/data/chemical/crop_images/{new_image_name}")
                image_info.update(
                    {
                        "original_image_path": f"/path/to/lvlm_evaluation/data_process/data/chemical/crop_images/{new_image_name}",
                        "category": category
                    }
                )

                self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv
        import copy

        num_choices = 4
        question = "What is the category of the chemical apparatus shown in the picture?"

        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "conical beaker",
                "question": question,
                "wrong_choices_list": ["erlenmeyer flask", "reagent bottle", "separatory funnel"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": dataset_info["category_space"][image_info["category"] - 1],
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
                
