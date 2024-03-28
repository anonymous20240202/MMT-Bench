from pathlib import Path
from collections import OrderedDict

from base_dataset import BaseDataset


class RAVEN_10000(BaseDataset):
    @staticmethod
    def process_raw_metadata_info(image_info):
        image_info["original_image_path"] = image_info["questions_images_path"][0]
        
        return image_info
    