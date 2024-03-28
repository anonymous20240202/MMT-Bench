from pathlib import Path


from base_dataset import BaseDataset


class am2k(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/image_matting/am-2k/validation/original",
        "trimap_path": "/path/to/lvlm_evaluation/data/image_matting/am-2k/validation/trimap",
        "mask_path": "/path/to/lvlm_evaluation/data/image_matting/am-2k/validation/mask",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        for image_path  in Path(self.image_path).iterdir():
            image_name = image_path.name

            original_image_path = str(image_path)

            trimap_path = str((Path(self.trimap_path) / image_name).with_suffix('.png'))
            mask_path = str((Path(self.mask_path) / image_name).with_suffix('.png'))

            image_info = self.image_dict("")
            image_info.update(
                {
                    "original_image_path": original_image_path,
                    "trimap_path": trimap_path,
                    "mask_path": mask_path
                }
            )

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv
        import random
        from pycocotools import mask as maskUtils
        import numpy as np
        from PIL import Image

        num_choices = 4

        width = image_info["width"]
        height = image_info["height"]
        # 使用 PIL 打开图像

        image = Image.open(image_info["mask_path"])
        alpha_map = np.array(image)

        if alpha_map.ndim > 2:
            alpha_map = alpha_map[:, :, 0]

        # print(alpha_map.shape)

        key = random.choices(["front", "back", "middle"], [2, 2, 8])[0]

        i = 0
        while i <= 10:
            try:
                if key == "front":
                    inside_indices = np.argwhere(alpha_map.reshape(-1) == 255)  # All indices where mask is nonzero
                    inside_index = random.choice(inside_indices) if len(inside_indices) > 0 else (0, 0)
                    inside_point = np.unravel_index(inside_index[0], alpha_map.shape[:2])

                    wrong_choices_list = [0]
                    wrong_choices_list.extend(random.sample(list(range(1, 255)), k=2))
                elif key == "back":
                    inside_indices = np.argwhere(alpha_map.reshape(-1) == 0)  # All indices where mask is nonzero
                    inside_index = random.choice(inside_indices) if len(inside_indices) > 0 else (0, 0)
                    inside_point = np.unravel_index(inside_index[0], alpha_map.shape[:2])

                    wrong_choices_list = [255]
                    wrong_choices_list.extend(random.sample(list(range(1, 255)), k=2))
                elif key == "middle":
                    inside_indices = np.argwhere((alpha_map.reshape(-1) > 0) & (alpha_map.reshape(-1) < 255))  # All indices where mask is nonzero
                    inside_index = random.choice(inside_indices) if len(inside_indices) > 0 else (0, 0)
                    inside_point = np.unravel_index(inside_index[0], alpha_map.shape[:2])

                    wrong_choices_list = [0, 255]
                    wrong_choices_list.extend(random.sample(list(range(1, 255)), k=1))
                
                alpha = alpha_map[inside_point]

                question = f"You are a professional image matting expert. What is the alpha value of the pixel point at coordinates {inside_point} in the image for image matting purposes? The alpha value represents the degree of transparency of the salient object against the background at this specific pixel. In this context, an alpha value of 0 indicates complete transparency, meaning the pixel is entirely invisible, while an alpha value of 255 represents complete opacity, meaning the pixel is fully visible. The dimensions of the input image are given as {width} in width and {height} in height. The coordinates of the top left corner of the image are (0, 0), and those of the bottom right corner are ({width}, {height})."

                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": alpha,
                    "question": question,
                    "wrong_choices_list": wrong_choices_list
                }

                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1
        
        return qa_json

    
class aim500(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/image_matting/aim-500/AIM-500/original",
        "trimap_path": "/path/to/lvlm_evaluation/data/image_matting/aim-500/AIM-500/trimap",
        "mask_path": "/path/to/lvlm_evaluation/data/image_matting/aim-500/AIM-500/mask",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        for image_path  in Path(self.image_path).iterdir():
            image_name = image_path.name

            original_image_path = str(image_path)

            trimap_path = str((Path(self.trimap_path) / image_name).with_suffix('.png'))
            mask_path = str((Path(self.mask_path) / image_name).with_suffix('.png'))

            image_info = self.image_dict("")
            image_info.update(
                {
                    "original_image_path": original_image_path,
                    "trimap_path": trimap_path,
                    "mask_path": mask_path
                }
            )
            
            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv
        import random
        from pycocotools import mask as maskUtils
        import numpy as np
        from PIL import Image

        num_choices = 4

        width = image_info["width"]
        height = image_info["height"]
        # 使用 PIL 打开图像

        image = Image.open(image_info["mask_path"])
        alpha_map = np.array(image)

        if alpha_map.ndim > 2:
            alpha_map = alpha_map[:, :, 0]

        # print(alpha_map.shape)

        key = random.choices(["front", "back", "middle"], [2, 2, 8])[0]

        i = 0
        while i <= 10:
            try:
                if key == "front":
                    inside_indices = np.argwhere(alpha_map.reshape(-1) == 255)  # All indices where mask is nonzero
                    inside_index = random.choice(inside_indices) if len(inside_indices) > 0 else (0, 0)
                    inside_point = np.unravel_index(inside_index[0], alpha_map.shape[:2])

                    wrong_choices_list = [0]
                    wrong_choices_list.extend(random.sample(list(range(1, 255)), k=2))
                elif key == "back":
                    inside_indices = np.argwhere(alpha_map.reshape(-1) == 0)  # All indices where mask is nonzero
                    inside_index = random.choice(inside_indices) if len(inside_indices) > 0 else (0, 0)
                    inside_point = np.unravel_index(inside_index[0], alpha_map.shape[:2])

                    wrong_choices_list = [255]
                    wrong_choices_list.extend(random.sample(list(range(1, 255)), k=2))
                elif key == "middle":
                    inside_indices = np.argwhere((alpha_map.reshape(-1) > 0) & (alpha_map.reshape(-1) < 255))  # All indices where mask is nonzero
                    inside_index = random.choice(inside_indices) if len(inside_indices) > 0 else (0, 0)
                    inside_point = np.unravel_index(inside_index[0], alpha_map.shape[:2])

                    wrong_choices_list = [0, 255]
                    wrong_choices_list.extend(random.sample(list(range(1, 255)), k=1))
                
                alpha = alpha_map[inside_point]

                question = f"You are a professional image matting expert. What is the alpha value of the pixel point at coordinates {inside_point} in the image for image matting purposes? The alpha value represents the degree of transparency of the salient object against the background at this specific pixel. In this context, an alpha value of 0 indicates complete transparency, meaning the pixel is entirely invisible, while an alpha value of 255 represents complete opacity, meaning the pixel is fully visible. The dimensions of the input image are given as {width} in width and {height} in height. The coordinates of the top left corner of the image are (0, 0), and those of the bottom right corner are ({width}, {height})."

                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": alpha,
                    "question": question,
                    "wrong_choices_list": wrong_choices_list
                }

                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1
        
        return qa_json
        