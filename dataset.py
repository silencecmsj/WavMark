import csv
from enum import Enum

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

import option


class Metadata(Enum):
    NAME = 0
    IMAGE = 1
    SPLIT = 2

class CHDD(Dataset):
    def __init__(self, args, state, transforms):
        self.csv_file_path = args.csv_file_path
        self.dataset = args.dataset
        self.state = state
        self._parse_list()
        self.transforms = transforms
        self.secret_images = Image.open(args.secret_image_path)

    def _parse_list(self):
        count = 0
        self.data = []
        with open(self.csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader.reader:
                if row[0] == 'name':
                    continue
                if self.state == 0 and row[Metadata.SPLIT.value] == 'train':
                    self.data.append(row)
                    count += 1
                if self.state == 1 and row[Metadata.SPLIT.value] == 'validate':
                    self.data.append(row)
                    count += 1
                if self.state == 2 and row[Metadata.SPLIT.value] == 'test':
                    self.data.append(row)
                    count += 1

        self.num = count

    def __getitem__(self, index):
        meta = self.data[index]
        name = meta[Metadata.NAME.value]

        raw_images = Image.open(meta[Metadata.IMAGE.value]).convert("RGB")
        raw_images = self.transforms(raw_images)
        secret_images = self.transforms(self.secret_images)

        return raw_images, secret_images, name

    def __len__(self):
        return len(self.data)