
import os
from PIL import Image
from torch.utils.data import Dataset

"""Script dealing with datasets of wafers patches"""

class WafwersDataSet(Dataset):
    def __init__(self, root_path: str, transform=None):
        self.root_path = root_path
        self.clean_image_names = os.listdir(os.path.join(self.root_path, "clean"))
        self.defect_image_names = os.listdir(os.path.join(self.root_path, "defect"))
        self.transform = transform

    def __len__(self):
        return len(self.clean_image_names) + len(self.defect_image_names)

    def __getitem__(self, index):
        """Returns an image patch and its label:
        1- defective
        0- clean
        """
        if index < self.__len__():
            all_images = self.defect_image_names + self.clean_image_names
            image_name = all_images[index]
            if os.path.exists(os.path.join(self.root_path, "defect", image_name)):
                label = 1
                image = Image.open(os.path.join(self.root_path, "defect", image_name))
            elif os.path.exists(os.path.join(self.root_path, "clean", image_name)):
                label = 0
                image = Image.open(os.path.join(self.root_path, "clean", image_name))
            if self.transform:
                image = self.transform(image)
            return (image, label)
