from pathlib import Path
import random

from base_dataset import BaseDataset

from tqdm import tqdm


class coco_pixel_recognition(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data/coco/val2017",
        "anno_path": "/path/to/lvlm_evaluation/data/coco/annotations/instances_val2017.json",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import mmcv
        from collections import defaultdict

        self.images_info = list()
        self.category_space = []
        anno_data_info = mmcv.load(self.anno_path)
        
        image2anno = defaultdict(list)

        for anno_info in anno_data_info["annotations"]:
            image2anno[anno_info["image_id"]].append(anno_info)

        id2category = dict()
        for cate_info in anno_data_info["categories"]:
            id2category[cate_info['id']] = cate_info["name"]
            self.category_space.append(cate_info["name"])
        
        for image_info in tqdm(anno_data_info["images"]):
            segment_list = image2anno[image_info["id"]]

            category_list = []
            segmentation_lsit = []

            flag = True
            for segment_info in segment_list:
                category = id2category[segment_info["category_id"]]
                segmentation = segment_info["segmentation"]

                if isinstance(segmentation, list):
                    if len(segmentation) > 1:
                        flag = False
                    else:
                        pass
                else:
                    flag = False


                category_list.append(category)
                segmentation_lsit.append(segmentation)
            
            if flag is False:
                continue

            if len(segmentation_lsit) >=4 or len(segmentation_lsit) == 0:
                continue
            
            image_path = Path(self.image_path) / image_info["file_name"]
            
            width, height = self.get_image_width_height(image_path)
            
            image_info = self.image_dict("")
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "segmentation_list": segmentation_lsit,
                    "category": category_list
                }
            )

            self.images_info.append(image_info)

        self.dataset_info["category_space"] = self.category_space
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv
        from pycocotools import mask as maskUtils
        import numpy as np

        def create_mask_from_polygon(polygon, image_height, image_width):
            """
            将COCO polygon坐标转换为二值mask。
            :param polygon: 多边形坐标列表。
            :param image_height: 图像的高度。
            :param image_width: 图像的宽度。
            :return: 二值mask。
            """
            rle = maskUtils.frPyObjects([polygon], image_height, image_width)
            mask = maskUtils.decode(rle)
            return mask

        def find_points_in_masks_with_random_outside(mask_list):
            """
            Find two points for each mask in the mask list: one inside the mask and one outside randomly.
            Each mask is assumed to be a numpy array of shape (h, w, 1).

            Args:
            mask_list (list of numpy.ndarray): List of masks, each of shape (h, w, 1).

            Returns:
            list of tuples: For each mask, a tuple of two points (inside_point, outside_point),
                            where each point is represented as (row, col).
            """
            points_list = []

            for mask in mask_list:
                # Find the index of a point inside the mask
                inside_indices = np.argwhere(mask.reshape(-1) > 0)  # All indices where mask is nonzero
                inside_index = random.choice(inside_indices) if len(inside_indices) > 0 else (0, 0)
                inside_point = np.unravel_index(inside_index[0], mask.shape[:2])

                # Find a random point outside the mask
                outside_indices = np.argwhere(mask.reshape(-1) == 0)  # All indices where mask is zero
                outside_index = random.choice(outside_indices) if len(outside_indices) > 0 else (0, 0)
                outside_point = np.unravel_index(outside_index[0], mask.shape[:2])

                points_list.append((inside_point, outside_point))

            return points_list

        num_choices = 4
        width = image_info["width"]
        height = image_info["height"]
        # question = f"What is the semantic category of the pixel point at coordinates {} in the image? For each detected object, provide the output in the format: category: ((x1, y1), (x2, y2)). The point (x1, y1) represents a coordinate on the detected object, and the point (x2, y2) represents a coordinate outside the detected object. Note that the width of the input image is given as {width} and the height as {height}."

        category_list = []
        mask_list = []

        for segmentation, category in zip(image_info["segmentation_list"], image_info["category"]):
            assert len(segmentation) == 1
            mask = create_mask_from_polygon(segmentation[0], height, width)

            category_list.append(category)
            mask_list.append(mask)
        
        # gt = find_points_in_masks_with_random_outside(mask_list)
        mask_id = random.choice(list(range(len(mask_list))))
        mask = mask_list[mask_id]
        category = category_list[mask_id]
        inside_indices = np.argwhere(mask.reshape(-1) > 0)  # All indices where mask is nonzero
        inside_index = random.choice(inside_indices) if len(inside_indices) > 0 else (0, 0)
        inside_point = np.unravel_index(inside_index[0], mask.shape[:2])


        num_choices = 4
        question = f"What is the semantic category of the pixel point at coordinates {inside_point} in the image? Note that the width of the input image is given as {width} and the height as {height}. The coordinates of the top left corner of the image are (0, 0), and the coordinates of the bottom right corner are ({width}, {height})."
        # question = "What type of animal is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "dog",
                "question": question,
                "wrong_choices_list": ["person", "train", "cat"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": category,
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)

        i = 0
        while i <= 10:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json


class ade20k_pixel_recognition(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/ade20k_/ade20k/images/validation",
        "anno_path": "/path/to/lvlm_evaluation/data_process/data/ade20k_/ade20k/annotations/validation",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import mmcv
        from collections import defaultdict

        self.images_info = list()
        self.category_space = []

        for image_path in Path(self.image_path).iterdir():
            anno_path = Path(self.anno_path) / f"{image_path.stem}.png"

            image_info = self.image_dict("")
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "mask_map_path": str(anno_path)
                }
            )

            self.images_info.append(image_info)

        category_info_list = mmcv.list_from_file("/path/to/lvlm_evaluation/data_process/data/ade20k_/ade20k/objectInfo150.txt")

        category_dict = dict()
        for category_info in category_info_list[1:]:
            category_id, _, _, _, category_name = category_info.split("\t")
            category_dict[int(category_id)] = category_name

        self.dataset_info["category_dict"] = category_dict
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv
        from pycocotools import mask as maskUtils
        import numpy as np
        from PIL import Image
        
        pass
        
        width = image_info["width"]
        height = image_info["height"]
        # 使用 PIL 打开图像
        image = Image.open(image_info["mask_map_path"])

        mask_map = np.array(image)

        inside_indices = np.argwhere(mask_map.reshape(-1) > 0)  # All indices where mask is nonzero
        inside_index = random.choice(inside_indices) if len(inside_indices) > 0 else (0, 0)
        inside_point = np.unravel_index(inside_index[0], mask_map.shape[:2])

        category_id = mask_map[inside_point]
        category = dataset_info["category_dict"][str(category_id)]

        num_choices = 4
        question = f"What is the semantic category of the pixel point at coordinates {inside_point} in the image? Note that the width of the input image is given as {width} and the height as {height}. The coordinates of the top left corner of the image are (0, 0), and the coordinates of the bottom right corner are ({width}, {height})."
        # question = "What type of animal is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": list(dataset_info["category_dict"].values()),
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "dog",
                "question": question,
                "wrong_choices_list": ["person", "train", "cat"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": category,
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)

        i = 0
        while i <= 10:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json
        





        

        # def create_mask_from_polygon(polygon, image_height, image_width):
        #     """
        #     将COCO polygon坐标转换为二值mask。
        #     :param polygon: 多边形坐标列表。
        #     :param image_height: 图像的高度。
        #     :param image_width: 图像的宽度。
        #     :return: 二值mask。
        #     """
        #     rle = maskUtils.frPyObjects([polygon], image_height, image_width)
        #     mask = maskUtils.decode(rle)
        #     return mask

        # def find_points_in_masks_with_random_outside(mask_list):
        #     """
        #     Find two points for each mask in the mask list: one inside the mask and one outside randomly.
        #     Each mask is assumed to be a numpy array of shape (h, w, 1).

        #     Args:
        #     mask_list (list of numpy.ndarray): List of masks, each of shape (h, w, 1).

        #     Returns:
        #     list of tuples: For each mask, a tuple of two points (inside_point, outside_point),
        #                     where each point is represented as (row, col).
        #     """
        #     points_list = []

        #     for mask in mask_list:
        #         # Find the index of a point inside the mask
        #         inside_indices = np.argwhere(mask.reshape(-1) > 0)  # All indices where mask is nonzero
        #         inside_index = random.choice(inside_indices) if len(inside_indices) > 0 else (0, 0)
        #         inside_point = np.unravel_index(inside_index[0], mask.shape[:2])

        #         # Find a random point outside the mask
        #         outside_indices = np.argwhere(mask.reshape(-1) == 0)  # All indices where mask is zero
        #         outside_index = random.choice(outside_indices) if len(outside_indices) > 0 else (0, 0)
        #         outside_point = np.unravel_index(outside_index[0], mask.shape[:2])

        #         points_list.append((inside_point, outside_point))

        #     return points_list

        # num_choices = 4
        # width = image_info["width"]
        # height = image_info["height"]
        # # question = f"What is the semantic category of the pixel point at coordinates {} in the image? For each detected object, provide the output in the format: category: ((x1, y1), (x2, y2)). The point (x1, y1) represents a coordinate on the detected object, and the point (x2, y2) represents a coordinate outside the detected object. Note that the width of the input image is given as {width} and the height as {height}."

        # category_list = []
        # mask_list = []

        # for segmentation, category in zip(image_info["segmentation_list"], image_info["category"]):
        #     assert len(segmentation) == 1
        #     mask = create_mask_from_polygon(segmentation[0], height, width)

        #     category_list.append(category)
        #     mask_list.append(mask)
        
        # # gt = find_points_in_masks_with_random_outside(mask_list)
        # mask_id = random.choice(list(range(len(mask_list))))
        # mask = mask_list[mask_id]
        # category = category_list[mask_id]
        # inside_indices = np.argwhere(mask.reshape(-1) > 0)  # All indices where mask is nonzero
        # inside_index = random.choice(inside_indices) if len(inside_indices) > 0 else (0, 0)
        # inside_point = np.unravel_index(inside_index[0], mask.shape[:2])


        # num_choices = 4
        # question = f"What is the semantic category of the pixel point at coordinates {inside_point} in the image? Note that the width of the input image is given as {width} and the height as {height}. The coordinates of the top left corner of the image are (0, 0), and the coordinates of the bottom right corner are ({width}, {height})."
        # question = "What type of animal is shown in the picture?"
        # sys_prompt = '\n'.join(mmcv.list_from_file('/path/to/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        # input_json = {
        #     "question": question,
        #     "category_space": dataset_info["category_space"],
        #     "example_dict": {
        #         "num_wrong_choices": num_choices - 1,
        #         "gt": "dog",
        #         "question": question,
        #         "wrong_choices_list": ["person", "train", "cat"]
        #     },
        #     "query_dict": {
        #         "num_wrong_choices": num_choices - 1,
        #         "gt": category,
        #         "question": question,
        #     }
        # }

        # user_prompt = json.dumps(input_json)

        # i = 0
        # while i <= 10:
        #     try:
        #         qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
        #         qa_json = BaseDataset.post_process(qa_json, question=question)
        #         break
        #     except:
        #         i += 1

        # return qa_json
