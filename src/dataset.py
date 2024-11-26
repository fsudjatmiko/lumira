import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class SimpleTorchDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, aug: list = []) -> None:
        self.dataset: list[tuple[str, np.ndarray]] = []
        self.root_dir = root_dir
        
        self.__add_dataset__("cables", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.__add_dataset__("case", [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.__add_dataset__("cpu", [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.__add_dataset__("gpu", [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.__add_dataset__("headset", [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.__add_dataset__("keyboard", [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.__add_dataset__("microphone", [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        self.__add_dataset__("monitor", [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        self.__add_dataset__("motherboard", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        self.__add_dataset__("mouse", [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        self.__add_dataset__("ram", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        self.__add_dataset__("speakers", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        self.__add_dataset__("storage", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        self.__add_dataset__("webcam", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        post_processing = [
            transforms.CenterCrop((128, 128)),
            transforms.ToTensor()
        ]

        self.augmentation = transforms.Compose(
            [transforms.Resize((150, 150))] + aug + post_processing
        )

    def __add_dataset__(self, dir_name: str, class_label: list[int]) -> None:
        full_path = os.path.join(self.root_dir, dir_name)
        label = np.array(class_label)
        for fname in os.listdir(full_path):
            fpath = os.path.join(full_path, fname)
            fpath = os.path.abspath(fpath)
            self.dataset.append((fpath, label))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        fpath, label = self.dataset[index]

        # load image into numpy RGB numpy array in pytorch format
        image = Image.open(fpath).convert('RGB')
        image = self.augmentation(image)

        # minmax norm the image
        image = (image - image.min()) / (image.max() - image.min())
        label = torch.Tensor(label)

        return image, label
