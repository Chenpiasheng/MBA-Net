import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from utils import transforms as T
from utils import transforms_edge as E
from torchvision.transforms import functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import random_split


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5, degrees=15):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        # trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomRotation(degrees),
            T.RandomCrop(crop_size),
            T.ToTensor(),
        ])

        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(imagesize=240):
    base_size = 565
    crop_size = imagesize
    return SegmentationPresetTrain(base_size, crop_size)


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=224, val_image_size=None, mode=""):
        """Initializes image paths and preprocessing module."""
        self.root = root
        self.image_size = image_size
        self.val_image_size = val_image_size

        self.flag = mode
        data_root = os.path.join(root, "DRIVE", self.flag)
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names]

    def __getitem__(self, idx):
        image_transforms = transforms.Compose([
            transforms.Grayscale(1)
        ])
        img = Image.open(self.img_list[idx]).convert('RGB')
        img = image_transforms(img)
        manual = Image.open(self.manual[idx]).convert('L')

        if self.flag == "train":
            image = np.array(img)
            label = np.array(manual) / 255

            image = Image.fromarray(image)
            label = Image.fromarray(label)

            transform = get_transform(self.image_size)

            image, label = transform(image, label)
            return image, label


        else:
            if self.val_image_size is not None:
                image_transforms = transforms.Compose([
                    transforms.CenterCrop(self.val_image_size),
                    transforms.ToTensor()
                ])
            else:
                image_transforms = transforms.Compose([
                    transforms.ToTensor()
                ])

            label = np.array(manual) / 255
            label = Image.fromarray(label)

            img = image_transforms(img)
            label = image_transforms(label)

            image = img.reshape(1, img.shape[1], img.shape[2])

        return image, label

    def __len__(self):
        return len(self.img_list)


def get_loader(image_path, image_size, val_image_size, batch_size, num_workers=0, mode='train', shuffle=True):
    dataset = ImageFolder(root=image_path, image_size=image_size, val_image_size=val_image_size, mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)
    return data_loader


if __name__ == "__main__":
    isbi_dataset = ImageFolder(r"../data", mode="train")
    print("number：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=1,
                                               shuffle=False)
    for image, label in train_loader:
        print(image)
