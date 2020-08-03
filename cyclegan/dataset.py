import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_A=None, transforms_B=None, dir_A="A", dir_B="B"):
        self.transform_A = transforms.Compose(transforms_A)
        self.transform_B = transforms.Compose(transforms_B)

        self.files_A = sorted(glob.glob(os.path.join(root, "%s" % dir_A) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s" % dir_B) + "/*.*"))
        print(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_B = Image.open(self.files_B[index % len(self.files_B)])

        item_A = self.transform_A(image_A)
        item_B = self.transform_B(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
