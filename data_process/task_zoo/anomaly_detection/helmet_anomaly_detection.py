from pathlib import Path

import mmcv

from base_dataset import BaseDataset

from PIL import Image

from tqdm import tqdm

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

        # Convert fractional coordinates to absolute pixel coordinates
        x, y, w, h = coordinates

        # Define the crop box
        crop_box = (x, y, x + w, y + h)

        # Crop and save the image
        cropped_img = img.crop(crop_box)
        cropped_img.save(save_path)


class helmet_anomaly_detection(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/lvlm_evaluation/Helmet_Anomaly_Detection/Helmet_Anomaly_Detection.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "The HELMET dataset contains 910 videoclips of motorcycle traffic, recorded at 12 observation sites in Myanmar in 2016. Each videoclip has a duration of 10 seconds, recorded with a framerate of 10fps and a resolution of 1920x1080. The dataset contains 10,006 individual motorcycles, surpassing the number of motorcycles available in existing datasets. Each motorcycle in the 91,000 annotated frames of the dataset is annotated with a bounding box, and rider number per motorcycle as well as position specific helmet use data is available."
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import os
        self.images_info = list()

        anno_info = mmcv.load(self.anno_path)

        for i, image_info in enumerate(tqdm(anno_info["images"])):

            if i > 1000:
                break

            for object_info in image_info["objects"]:

                if object_info["label"] not in ["DNoHelmet", "DHelmet"]:
                    continue

                x, y, w, h = int(object_info["x"]), int(object_info["y"]), int(object_info["w"]), int(object_info["h"])

                new_image_name = self.new_image_name()
                BaseDataset.exist_or_mkdir("/path/to/lvlm_evaluation/data_process/taskonomy_evaluation_data/anomaly_detection/helmet_anomaly_detection/images/")
                crop_and_save_image_center(image_info["original_image_path"], (x, y, w, h), f"/path/to/lvlm_evaluation/data_process/taskonomy_evaluation_data/anomaly_detection/helmet_anomaly_detection/images/{new_image_name}")

                _image_info = self.image_dict("")
                _image_info.update(
                    {
                        "original_image_path": os.path.join("/path/to/lvlm_evaluation/data_process/taskonomy_evaluation_data/anomaly_detection/helmet_anomaly_detection/images/", new_image_name),
                        "category": object_info["label"]
                    }
                )

                self.images_info.append(_image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        dict_h = {
            "DHelmet": "Yes",
            "DHelmetP1NoHelmet": "No"
        }

        num_choices = 2
        question = f"Is the person in the picture wearing a helmet?"
        category = image_info["category"]

        if category == "DHelmet":
            gt = "Yes"
            wrong_choices_list = ["No"]
        elif category == "DNoHelmet":
            gt = "No"
            wrong_choices_list = ["Yes"]
        else:
            raise NotImplementedError

        qa_json = {
            "num_wrong_choices": num_choices - 1,
            "gt": gt,
            "question": question,
            "wrong_choices_list": wrong_choices_list
        }

        qa_json = BaseDataset.post_process(qa_json, question)

        return qa_json
