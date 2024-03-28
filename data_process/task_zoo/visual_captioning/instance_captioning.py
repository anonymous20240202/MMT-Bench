import os 
from pathlib import Path

import mmcv
import numpy as np
from base_dataset import BaseDataset

from tqdm import tqdm

from mmdet.datasets.api_wrappers import COCO


# def mark_image_to_base64(img, target_size=-1):
#     # if target_size == -1, will not do resizing
#     # else, will set the max_size ot (target_size, target_size)
#     if img.mode in ("RGBA", "P"):
#         img = img.convert("RGB")
#     tmp = osp.join('/tmp', str(uuid4()) + '.jpg')
#     if target_size > 0:
#         img.thumbnail((target_size, target_size))
#     img.save(tmp)
#     ret = encode_image_file_to_base64(tmp)
#     os.remove(tmp)
#     return ret


class visual_genome_caption(BaseDataset):
    CLASSES = ('object',)
    DATA_METAINFO = {
        "anno_path": "/path/to/metainfo/instance.json",
        "img_prefix": "/path/to/visual_genome/vg_all",
        "save_image_path": "",
        "sampling_num": 100,
        "visual_input_component": ["natural_image", ],
        "dataset_description": "a picture may contains multiple regions and multiple instances, the caption is made towards a certain instance/region"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    
    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            # convert data type for flickr
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])

            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        # flickr
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                # flickr label
                # TODO
                gt_labels.append(ann['caption'])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            file_name=img_info['filename'])

        return ann
    
    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)
    
    
    def parse_images_info(self):
        self.data_infos = self.load_annotations("/path/to/lvlm_evaluation/data_process/data/instance_region/test.json")
        self.images_info = list()

        for i in tqdm(range(len(self.data_infos))):
            data_info = self.get_ann_info(i)

            num_bboxes = data_info["bboxes"].shape[0]

            for j in range(num_bboxes):

                image_info = self.image_dict("")
                bboxes = data_info["bboxes"][j].tolist()
                bboxes = [round(_) for _ in bboxes]

                caption = data_info["labels"][j]

                image_info.update(
                    {
                        "original_image_path": os.path.join(self.img_prefix, data_info["file_name"]),
                        "bbox": bboxes,
                        "caption": caption
                    }
                )

                self.images_info.append(image_info)

    # @staticmethod
    # def generate_qa(image_info, dataset_info, save_image_path):
    #     import mmcv
    #     import numpy as np
    #     import json
    #     import random
    #     import json
    #     import mmcv
    #     from openai import OpenAI
    #     from prompt.utils import encode_image_to_base64
    #     num_choices = 4

    #     # question = image_info["question"]
    #     # gt = image_info["gt_answer"]
    #     question = "Describe this image briefly."
    #     caption = image_info["caption"]

    #     gpt4v_prompt = f"Based on the image and the correct caption I provided, generate three incorrect captions that are misleading.\nCorrect Caption: {caption}"
    #     chatgpt_prompt = "Your task is to extract options from the provided text and return them in JSON format, encapsulated within a Python dictionary. This dictionary should contain only one key, options, whose corresponding value is a list, with each element of the list being an option. Please do not include option symbols such as A, B, C in the list elements."

    #     # Path to your image
    #     # image_path = image_info["original_image_path"]

    #     from PIL import Image
    #     # Getting the base64 string
    #     base64_image = encode_image_to_base64(image_path, 768)
    #     BaseDataset.exist_or_mkdir(save_image_path)

    #     bbox_list = [np.array([image_info['bbox']])]
    #     tmp_image_path = str(Path('/tmp') / BaseDataset.new_image_name())
    #     save_image_path = str(Path(save_image_path) / BaseDataset.new_image_name())
    #     mmcv.imshow_bboxes(image_info["original_image_path"], bbox_list, show=False, out_file=save_image_path, thickness=2, colors='red')

    #     image_path = save_image_path

    #     image_path["_original_image_path"] = image_info["original_image_path"]


    #     openai_client = OpenAI()
    #     response = openai_client.chat.completions.create(
    #         model="gpt-4-vision-preview",
    #         messages=[
    #             {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": gpt4v_prompt},
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{base64_image}",
    #                         "image_detail": "low"
    #                     }
    #                 },
    #             ],
    #             }
    #         ],
    #         max_tokens=300,
    #     )

    #     i = 0
    #     while i <= 10:
    #         try:
    #             options = BaseDataset.openai_generate(chatgpt_prompt, response.choices[0].message.content)["options"]
    #             qa_json = {
    #                 "wrong_choices_list": options,
    #                 "question": question,
    #                 "num_wrong_choices": len(options),
    #                 "gt": caption,
    #             }
    #             qa_json = BaseDataset.post_process(qa_json, question=question)
    #             qa_json["original_image_path"] = save_image_path
    #             break
    #         except:
    #             i += 1

    #     return qa_json
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import mmcv
        import numpy as np
        import json
        import random
        import json
        import mmcv
        from openai import OpenAI
        from prompt.utils import encode_image_to_base64
        num_choices = 4

        # question = image_info["question"]
        # gt = image_info["gt_answer"]
        question = "Describe the region (marked as RED) in image briefly."
        caption = image_info["caption"]

        gpt4v_prompt = f"I will provide you with an image in which one area is marked with a red box. Additionally, I will give you the correct caption for that area. Please generate three incorrect captions for that area. The generated captions should be misleading while still being related to the content of the image.\nCorrect Caption: {caption}"
        chatgpt_prompt = "Your task is to extract options from the provided text and return them in JSON format, encapsulated within a Python dictionary. This dictionary should contain only one key, options, whose corresponding value is a list, with each element of the list being an option. Please do not include option symbols such as A, B, C in the list elements."


        from PIL import Image
        BaseDataset.exist_or_mkdir(save_image_path)

        bbox_list = [np.array([image_info['bbox']])]
        tmp_image_path = str(Path('/tmp') / BaseDataset.new_image_name())
        save_image_path = str(Path(save_image_path) / BaseDataset.new_image_name())
        mmcv.imshow_bboxes(image_info["original_image_path"], bbox_list, show=False, out_file=save_image_path, thickness=2, colors='red')

        image_path = save_image_path

        image_info["_original_image_path"] = image_info["original_image_path"]
        # Path to your image

        from PIL import Image
        # Getting the base64 string
        base64_image = encode_image_to_base64(Image.open(image_path), 768)

        openai_client = OpenAI()
        response = openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": gpt4v_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "image_detail": "low"
                        }
                    },
                ],
                }
            ],
            max_tokens=300,
        )

        options = BaseDataset.openai_generate(chatgpt_prompt, response.choices[0].message.content)["options"]

        i = 0
        while i <= 10:
            try:
                options = BaseDataset.openai_generate(chatgpt_prompt, response.choices[0].message.content)["options"][:4]
                qa_json = {
                    "wrong_choices_list": options,
                    "question": question,
                    "num_wrong_choices": len(options),
                    "gt": caption,
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                qa_json["original_image_path"] = save_image_path
                break
            except:
                i += 1

        return qa_json

class refcocog_caption(BaseDataset):
    CLASSES = ('object',)
    DATA_METAINFO = {
        "anno_path": "/path/to/metainfo/instance.json",
        "img_prefix": "/path/to/lvlm_evaluation/data_process/data/coco/train2014",
        "sampling_num": 100,
        "visual_input_component": ["natural_image", ],
        "dataset_description": "a picture may contains multiple regions and multiple instances, the caption is made towards a certain instance/region"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        num_remove_images = 0
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            if len(info['caption'].split(' ')) < 3:
                num_remove_images += 1
                continue
            info['filename'] = info['file_name']
            # info['filename'] = info['file_name'].split('_')[-1]
            # convert data type for flickr
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])

            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        print(f'Filtered {num_remove_images} from  {ann_file} ')
        self.id_cap_dict = {}
        return data_infos
    
    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        img_path = os.path.join(self.img_prefix, img_info['file_name'].split('_')[-1])
        self.id_cap_dict[img_info['file_name'].split('_')[-1]] = img_info['caption']
        # flickr
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue

            bbox = [x1, y1, x1 + w, y1 + h]

            gt_bboxes.append(bbox)
            gt_labels.append(img_info['caption'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)

        #mmcv.imshow_bboxes(img_path, gt_bboxes, win_name=img_info['caption'])

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            caption=img_info['caption'],
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map, 
            file_path = img_info['file_name'])

        return ann

    def process_text(self, data_item):
        if isinstance(data_item['img'], list):
            # test model
            data_item = {k: v[0] for k, v in data_item.items()}

        return self.train_process_test(data_item)

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def parse_images_info(self):
        self.data_infos = self.load_annotations("/path/to/lvlm_evaluation/data_process/data/instance_region/finetune_refcocog_val.json")
        self.images_info = list()

        for i in tqdm(range(len(self.data_infos))):
            data_info = self.get_ann_info(i)

            num_bboxes = data_info["bboxes"].shape[0]

            for j in range(num_bboxes):

                image_info = self.image_dict("")
                bboxes = data_info["bboxes"][j].tolist()
                bboxes = [round(_) for _ in bboxes]

                caption = data_info["labels"][j]

                image_info.update(
                    {
                        "original_image_path": os.path.join(self.img_prefix, data_info["file_path"]),
                        "bbox": bboxes,
                        "caption": caption
                    }
                )

                self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import mmcv
        import numpy as np
        import json
        import random
        import json
        import mmcv
        from openai import OpenAI
        from prompt.utils import encode_image_to_base64
        num_choices = 4

        # question = image_info["question"]
        # gt = image_info["gt_answer"]
        question = "Describe the region (marked as RED) in image briefly."
        caption = image_info["caption"]

        gpt4v_prompt = f"I will provide you with an image in which one area is marked with a red box. Additionally, I will give you the correct caption for that area. Please generate three incorrect captions for that area. The generated captions should be misleading while still being related to the content of the image.\nCorrect Caption: {caption}"
        chatgpt_prompt = "Your task is to extract options from the provided text and return them in JSON format, encapsulated within a Python dictionary. This dictionary should contain only one key, options, whose corresponding value is a list, with each element of the list being an option. Please do not include option symbols such as A, B, C in the list elements."


        from PIL import Image
        BaseDataset.exist_or_mkdir(save_image_path)

        bbox_list = [np.array([image_info['bbox']])]
        tmp_image_path = str(Path('/tmp') / BaseDataset.new_image_name())
        save_image_path = str(Path(save_image_path) / BaseDataset.new_image_name())
        mmcv.imshow_bboxes(image_info["original_image_path"], bbox_list, show=False, out_file=save_image_path, thickness=2, colors='red')

        image_path = save_image_path

        image_info["_original_image_path"] = image_info["original_image_path"]
        # Path to your image

        from PIL import Image
        # Getting the base64 string
        base64_image = encode_image_to_base64(Image.open(image_path), 768)

        openai_client = OpenAI()
        response = openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": gpt4v_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "image_detail": "low"
                        }
                    },
                ],
                }
            ],
            max_tokens=300,
        )

        options = BaseDataset.openai_generate(chatgpt_prompt, response.choices[0].message.content)["options"]

        i = 0
        while i <= 10:
            try:
                options = BaseDataset.openai_generate(chatgpt_prompt, response.choices[0].message.content)["options"][:4]
                qa_json = {
                    "wrong_choices_list": options,
                    "question": question,
                    "num_wrong_choices": len(options),
                    "gt": caption,
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                qa_json["original_image_path"] = save_image_path
                break
            except:
                i += 1

        return qa_json
