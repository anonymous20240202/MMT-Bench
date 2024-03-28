from pathlib import Path
import copy
import random

import mmcv
import numpy as np
from iso3166 import countries

from base_dataset import BaseDataset


import cv2
import numpy as np
from tqdm import tqdm

from prompt.utils import *


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def plot_and_save_two_images(image_path1, image_path2, label1, label2, save_path, dpi):
    """
    Plots two images side by side with their respective labels and saves the plot to a specified path.

    Parameters:
    image_path1 (str): File path of the first image.
    image_path2 (str): File path of the second image.
    label1 (str): Label for the first image.
    label2 (str): Label for the second image.
    save_path (str): Path to save the combined image plot.
    """

    # 读取两张图片
    img1 = mpimg.imread(image_path1)
    img2 = mpimg.imread(image_path2)

    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # 在子图上显示图片和标签
    ax1.imshow(img1)
    ax1.set_title(label1)
    ax1.axis('off')  # 关闭坐标轴

    ax2.imshow(img2)
    ax2.set_title(label2)
    ax2.axis('off')  # 关闭坐标轴

    # 调整子图之间的间隔
    plt.subplots_adjust(wspace=0.2)

    # 保存图形到指定路径
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)


def mask_to_box(mask_path):
    # 读取 mask 图像
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 寻找 mask 中的非零像素点
    y_indices, x_indices = np.where(mask > 0)
    
    # 计算 bounding box 的坐标
    x_min, x_max = min(x_indices), max(x_indices)
    y_min, y_max = min(y_indices), max(y_indices)
    
    # 返回 bounding box 的 (x, y, width, height)
    return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))


def get_answer(gt, rangee=4, num=3):
    import random
    a = list(range(max(0, gt - rangee), gt + rangee))
    a.remove(gt)
    return random.sample(a, k=num)


class fss_1000(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/path/to/lvlm_evaluation/data_process/data/perdet/fss_1000/FSS-1000",
        "sampling_num": 100,
        "visual_input_component": ["natural_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        # anno_info_lsit = mmcv.load(self.anno_path)
        for image_category_path in tqdm(Path(self.image_path).iterdir()):
            support_image_list = []
            support_anno_list = []
            id_list = []
            for i, image_path in enumerate(image_category_path.iterdir()):
                if image_path.suffix == ".png":
                    continue
                anno_path = image_path.with_suffix(".png")

                support_image_list.append(image_path)
                support_anno_list.append(anno_path)
                id_list.append(i)
            for i, image_path in enumerate(image_category_path.iterdir()):
                if image_path.suffix == ".png":
                    continue
                anno_path = image_path.with_suffix(".png")
                _id_list = copy.deepcopy(id_list)
                # _id_list = list(range(len(support_image_list)))
                _id_list.remove(i)
                sup_id = random.choice(_id_list)
                sup_id = _id_list.index(sup_id)
                support_image_path = support_image_list[sup_id]
                support_anno_path = support_anno_list[sup_id]

                support_bbox_coordinates = mask_to_box(str(support_anno_path))
                query_bbox_coordinates = mask_to_box(str(anno_path))

                image_info = self.image_dict("")
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "query_bbox_coordinates": query_bbox_coordinates,
                        "support_image_path": str(support_image_path),
                        "support_bbox_coordinates": support_bbox_coordinates,
                    }
                )
                self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import json
        import mmcv

        num_choices = 4
        width, height = image_info["width"], image_info["height"]
        question = f"According to the prompts in the Support Image (marked in red), please detect the corresponding object in the Query Image. Provide the output for the object in the format [x, y, w, h]. This format represents the bounding box, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as its width and height. Note that the width of the input RGB image is {width} and the height is {height}."

        bbox = image_info["support_bbox_coordinates"]
        gt_bbox = image_info["query_bbox_coordinates"]

        BaseDataset.exist_or_mkdir(save_image_path)

        bbox_list = [np.array([np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])])]
        new_image_path = str(Path(save_image_path) / BaseDataset.new_image_name())
        mmcv.imshow_bboxes(image_info["support_image_path"], bbox_list, show=False, out_file=new_image_path, thickness=2, colors="red")

        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_two_images(new_image_path, image_info["original_image_path"], "Support Image", "Query Image", merge_image_path, dpi=200)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_bounding_box_from_single_bbox(gt_bbox, width, height, num_choices - 1)
                    
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt_bbox,
                    "question": question,
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                qa_json["merge_image_path"] = merge_image_path

                qa_json["original_image_name"] = ["Support Image", "Query Image"]
                break
            except:
                i += 1

        return qa_json


r""" PACO-Part few-shot semantic segmentation dataset """
import os
import pickle

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from detectron2.structures.masks import *

class DatasetPACOPart(Dataset):
    def __init__(self, datapath, fold=1, transform=None, split='val', shot=1, use_original_imgsize=False, box_crop=True):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 448
        self.benchmark = 'paco_part'
        self.shot = shot
        self.img_path = "/path/to/lvlm_evaluation/data_process/data/coco"
        self.anno_path = os.path.join(datapath, 'paco_part', 'paco')
        self.transform = transforms.Compose([
            transforms.Resize(size=(512, 512)),
            transforms.ToTensor()
        ])
        self.use_original_imgsize = use_original_imgsize
        self.box_crop = box_crop

        self.class_ids_ori, self.cid2img, self.img2anno = self.build_img_metadata_classwise()
        self.class_ids_c = {cid: i for i, cid in enumerate(self.class_ids_ori)}
        self.class_ids = sorted(list(self.class_ids_c.values()))
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 2500

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()

        query_img = self.transform(query_img)
        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
        support_masks = torch.stack(support_masks)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'org_query_imsize': org_qry_imsize,
                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'class_id': torch.tensor(self.class_ids_c[class_sample])}

        return batch

    def build_img_metadata_classwise(self):

        with open(os.path.join(self.anno_path, 'paco_part_train.pkl'), 'rb') as f:
            train_anno = pickle.load(f)
        with open(os.path.join(self.anno_path, 'paco_part_val.pkl'), 'rb') as f:
            test_anno = pickle.load(f)

        # Remove Duplicates
        new_cid2img = {}

        for cid_id in test_anno['cid2img']:
            id_list = []
            if cid_id not in new_cid2img:
                new_cid2img[cid_id] = []
            for img in test_anno['cid2img'][cid_id]:
                img_id = list(img.keys())[0]
                if img_id not in id_list:
                    id_list.append(img_id)
                    new_cid2img[cid_id].append(img)
        test_anno['cid2img'] = new_cid2img

        train_cat_ids = list(train_anno['cid2img'].keys())
        test_cat_ids = [i for i in list(test_anno['cid2img'].keys()) if len(test_anno['cid2img'][i]) > self.shot]
        assert len(train_cat_ids) == self.nclass

        nclass_trn = self.nclass // self.nfolds

        class_ids_val = [train_cat_ids[self.fold + self.nfolds * v] for v in range(nclass_trn)]
        class_ids_val = [x for x in class_ids_val if x in test_cat_ids]
        class_ids_trn = [x for x in train_cat_ids if x not in class_ids_val]

        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val
        img_metadata_classwise = train_anno if self.split == 'trn' else test_anno
        cid2img = img_metadata_classwise['cid2img']
        img2anno = img_metadata_classwise['img2anno']

        return class_ids, cid2img, img2anno

    def build_img_metadata(self):
        img_metadata = []
        for k in self.cid2img.keys():
            img_metadata += self.cid2img[k]
        return img_metadata

    def get_mask(self, segm, image_size):

        if isinstance(segm, list):
            # polygon
            polygons = [np.asarray(p) for p in segm]
            mask = polygons_to_bitmask(polygons, *image_size[::-1])
        elif isinstance(segm, dict):
            # COCO RLE
            mask = mask_util.decode(segm)
        elif isinstance(segm, np.ndarray):
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            mask = segm
        else:
            raise NotImplementedError

        return torch.tensor(mask)


    def load_frame(self):
        class_sample = np.random.choice(self.class_ids_ori, 1, replace=False)[0]
        query = np.random.choice(self.cid2img[class_sample], 1, replace=False)[0]
        query_id, query_name = list(query.keys())[0], list(query.values())[0]
        query_name = '/'.join( query_name.split('/')[-2:])
        query_img = Image.open(os.path.join(self.img_path, query_name)).convert('RGB')
        org_qry_imsize = query_img.size
        query_annos = self.img2anno[query_id]

        query_obj_dict = {}

        for anno in query_annos:
            if anno['category_id'] == class_sample:
                obj_id = anno['obj_ann_id']
                if obj_id not in query_obj_dict:
                    query_obj_dict[obj_id] = {
                        'obj_bbox': [],
                        'segms': []
                    }
                query_obj_dict[obj_id]['obj_bbox'].append(anno['obj_bbox'])
                query_obj_dict[obj_id]['segms'].append(self.get_mask(anno['segmentation'], org_qry_imsize)[None, ...])

        sel_query_id = np.random.choice(list(query_obj_dict.keys()), 1, replace=False)[0]
        query_obj_bbox = query_obj_dict[sel_query_id]['obj_bbox'][0]
        query_part_masks = query_obj_dict[sel_query_id]['segms']
        query_mask = torch.cat(query_part_masks, dim=0)
        query_mask = query_mask.sum(0) > 0

        support_names = []
        support_pre_masks = []
        support_boxes = []
        while True:  # keep sampling support set if query == support
            support = np.random.choice(self.cid2img[class_sample], 1, replace=False)[0]
            support_id, support_name = list(support.keys())[0], list(support.values())[0]
            support_name = '/'.join(support_name.split('/')[-2:])
            if query_name != support_name:
                support_names.append(support_name)
                support_annos = self.img2anno[support_id]

                support_obj_dict = {}
                for anno in support_annos:
                    if anno['category_id'] == class_sample:
                        obj_id = anno['obj_ann_id']
                        if obj_id not in support_obj_dict:
                            support_obj_dict[obj_id] = {
                                'obj_bbox': [],
                                'segms': []
                            }
                        support_obj_dict[obj_id]['obj_bbox'].append(anno['obj_bbox'])
                        support_obj_dict[obj_id]['segms'].append(anno['segmentation'])

                sel_support_id = np.random.choice(list(support_obj_dict.keys()), 1, replace=False)[0]
                support_obj_bbox = support_obj_dict[sel_support_id]['obj_bbox'][0]
                support_part_masks = support_obj_dict[sel_support_id]['segms']

                support_boxes.append(support_obj_bbox)
                support_pre_masks.append(support_part_masks)

            if len(support_names) == self.shot:
                break

        support_imgs = []
        support_masks = []
        for support_name, support_pre_mask in zip(support_names, support_pre_masks):
            support_img = Image.open(os.path.join(self.img_path, support_name)).convert('RGB')
            support_imgs.append(support_img)
            org_sup_imsize = support_img.size
            sup_masks = []
            for pre_mask in support_pre_mask:
                sup_masks.append(self.get_mask(pre_mask, org_sup_imsize)[None, ...])
            support_mask = torch.cat(sup_masks, dim=0)
            support_mask = support_mask.sum(0) > 0

            support_masks.append(support_mask)

        if self.box_crop:
            query_img = np.asarray(query_img)
            query_img = query_img[int(query_obj_bbox[1]):int(query_obj_bbox[1]+query_obj_bbox[3]), int(query_obj_bbox[0]):int(query_obj_bbox[0]+query_obj_bbox[2])]
            query_img = Image.fromarray(np.uint8(query_img))
            org_qry_imsize = query_img.size
            query_mask = query_mask[int(query_obj_bbox[1]):int(query_obj_bbox[1]+query_obj_bbox[3]), int(query_obj_bbox[0]):int(query_obj_bbox[0]+query_obj_bbox[2])]

            new_support_imgs = []
            new_support_masks = []

            for sup_img, sup_mask, sup_box in zip(support_imgs, support_masks, support_boxes):
                sup_img = np.asarray(sup_img)
                sup_img = sup_img[int(sup_box[1]):int(sup_box[1]+sup_box[3]), int(sup_box[0]):int(sup_box[0]+sup_box[2])]
                sup_img = Image.fromarray(np.uint8(sup_img))

                new_support_imgs.append(sup_img)
                new_support_masks.append(sup_mask[int(sup_box[1]):int(sup_box[1]+sup_box[3]), int(sup_box[0]):int(sup_box[0]+sup_box[2])])

            support_imgs = new_support_imgs
            support_masks = new_support_masks

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize


      
r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader
class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            'paco_part': DatasetPACOPart,
        }

        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor()
        ])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader

from scipy import ndimage
def count_connected_regions(mask):
    # 找到连通区域
    labeled, num_features = ndimage.label(mask)
    
    # 计算每个连通区域的面积
    region_sizes = np.bincount(labeled.flatten())

    # 设置小区域的标签为 0（不考虑）
    labeled[np.isin(labeled, np.where(region_sizes < 16))] = 0

    # 重新计算连通区域个数（排除小于 16 的区域）
    labeled[labeled > 0] = 1
    labeled, num_features = ndimage.label(labeled)
    
    return num_features

def count_connected_components(mask_array):
    # 确保 mask 是二值图像
    ret, binary_mask = cv2.threshold(mask_array, 0, 255, cv2.THRESH_BINARY)

    # 应用 connectedComponents
    num_labels, labels = cv2.connectedComponents(binary_mask)

    # num_labels 包含背景，所以连通域的数量需要减一
    return num_labels - 1


def mask2bbox(mask):
    # 获取所有非零元素的行索引和列索引
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # 找出最小和最大的行索引和列索引
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # 返回边界框坐标
    return int(x_min), int(y_min), int(x_max), int(y_max)


class paco_part(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/path/to/lvlm_evaluation/data_process/data/perdet/paco_part/paco/paco_part_val.pkl",
        "save_image_path": "/path/to/lvlm_evaluation/data_process/taskonomy_evaluation_data/cross_image_matching/one_shot_detection/images",
        "image_path": "/path/to/lvlm_evaluation/data_process/data/coco/val2017",
        "sampling_num": 100,
        "visual_input_component": ["natural_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        paco_part_data_info = DatasetPACOPart(datapath="/path/to/lvlm_evaluation/data_process/data/perdet")
        i = 0
        # anno_info_lsit = mmcv.load(self.anno_path)
        bar = tqdm(400)
        for sample_info in paco_part_data_info:
            query_img = sample_info["query_img"]
            query_mask = sample_info["query_mask"]
            support_img = sample_info["support_imgs"][0]
            support_mask = sample_info["support_masks"][0]

            if count_connected_regions(query_mask) == 1 and count_connected_regions(support_mask) == 1:
                pass
            else:
                continue

            query_image_path = os.path.join(self.save_image_path, self.new_image_name())
            support_image_path = os.path.join(self.save_image_path, self.new_image_name())

            self.save_rgb_image((query_img.permute(1, 2, 0).numpy()*255)[:, :, ::-1], query_image_path)
            self.save_rgb_image((support_img.permute(1, 2, 0).numpy()*255)[:, :, ::-1], support_image_path)

            query_bbox_coordinates = mask2bbox(query_img.numpy())
            support_bbox_coordinates = mask2bbox(support_img.numpy())

            image_info = self.image_dict("")
            image_info.update(
                {
                    "original_image_path": str(query_image_path),
                    "query_bbox_coordinates": query_bbox_coordinates,
                    "support_image_path": str(support_image_path),
                    "support_bbox_coordinates": support_bbox_coordinates,
                }
            )
            self.images_info.append(image_info)
            bar.update(1)

            if len(self.images_info) >= 400:
                break
    

    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import json
        import mmcv

        num_choices = 4
        width, height = image_info["width"], image_info["height"]
        question = f"According to the prompts in the Support Image (marked in red), please detect the corresponding object in the Query Image. Provide the output for the object in the format [x, y, w, h]. This format represents the bounding box, where [x, y, w, h] are the coordinates of the top-left corner of the bounding box, as well as its width and height. Note that the width of the input RGB image is {width} and the height is {height}."

        bbox = image_info["support_bbox_coordinates"]
        gt_bbox = image_info["query_bbox_coordinates"]

        BaseDataset.exist_or_mkdir(save_image_path)

        bbox_list = [np.array([np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])])]
        new_image_path = str(Path(save_image_path) / BaseDataset.new_image_name())
        mmcv.imshow_bboxes(image_info["support_image_path"], bbox_list, show=False, out_file=new_image_path, thickness=2, colors="red")

        merge_image_path = os.path.join(save_image_path, BaseDataset.new_image_name())

        plot_and_save_two_images(new_image_path, image_info["original_image_path"], "Support Image", "Query Image", merge_image_path, dpi=200)

        i = 0
        while i <= 10:
            try:
                wrong_choices_list = generate_incorrect_bounding_box_from_single_bbox(gt_bbox, width, height, num_choices - 1)
                    
                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": gt_bbox,
                    "question": question,
                    "wrong_choices_list": wrong_choices_list
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                qa_json["merge_image_path"] = merge_image_path

                qa_json["original_image_name"] = ["Support Image", "Query Image"]
                break
            except:
                i += 1

        return qa_json