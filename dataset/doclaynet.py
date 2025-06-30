"""
DocLayNet dataset
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from datasets import load_dataset
import torchvision.transforms.functional

import dataset.transforms as T


class DocLayNet(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms):
        self._transforms = transforms
        self.dataset = dataset

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]

        w, h = image.size

        classes = torch.tensor(item["category_id"], dtype=torch.int64)
        boxes = item["bboxes"]

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {'image_id': torch.tensor([idx]), 
                  'boxes': boxes, 
                  'labels': classes}
        
        area = torch.tensor(item["area"])
        iscrowd = torch.tensor([0 for obj in item["category_id"]])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self._transforms is not None:
            img, target = self._transforms(image, target)
        return img, target
    
    def __len__(self):
        return len(self.dataset)


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            #T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    dataset = load_dataset("ds4sd/DocLayNet-v1.1", split=image_set)
    dataset = DocLayNet(dataset, transforms=make_coco_transforms(image_set))
    return dataset
