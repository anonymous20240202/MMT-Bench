from pathlib import Path
from collections import OrderedDict

from base_dataset import BaseDataset


class kinetics400(BaseDataset):
    @staticmethod
    def process_raw_metadata_info(image_info):
        if len(image_info["original_image_path"]):
            return None
        else:
            return image_info
    