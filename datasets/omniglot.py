import torch
import torch.utils.data as data
from torchvision.datasets import Omniglot
from torchvision.transforms import ToTensor

import glob
import pickle
from PIL import Image
import numpy as np
import os


OMNIGLOT_TEST_CLASSES = ["Bengali", "Ge_ez", "Glagolitic", "Gurmukhi", "Kannada",
                         "Malayalam", "Manipuri", "Old_Church_Slavonic_(Cyrillic)", "Oriya", "Tibetan"]


class RestrictedOmniglot(data.Dataset):

    def __init__(self, root, n_classes, train=True, download=True, noise_std=0):
        """
        There is already a realization of Omniglot in torchvision.datasets, main differences of this version are:
        * Dataset considers only a subset of all claases at each time and may reshuffle classes
        * Different breakdown of characters into training and validation
          Check https://github.com/syed-ahmed/MatchingNetworks-OSL for more details
        * Adding additional classes by rotating images by 90, 180 and 270 degrees
        * All data is preloaded into memory - because of that no option for image and target transforms
          - instead we hardcode .Resize() and toTensor() transforms
        * Adding noise augmentation for training set
        Args:
            root: root directory of images as in torchvision.datasets.Omniglot
            n_classes: total number of claases considered
            train: training or validation dataset
            download: flag to download the data
            noise_std: standard deviation of Gaussian noise applied to training data
        """
        super(RestrictedOmniglot, self).__init__()
        self.root = root
        self.n_classes = n_classes
        self.train = train
        self.download = download
        self.noise_std = noise_std

        # Hardcoding some dataset settings
        self.rotations = [0, 90, 180, 270]
        self.images_per_class = 20
        self.image_size = 28

        cache_pkl = os.path.join(self.root, 'omniglot_packed.pkl')
        try:
            with open(cache_pkl, 'rb') as f:
                class_members = pickle.load(f)
            self.data, self.target, self.n_all_classes, self.target_mapping = class_members
        except IOError:
            # Relying on pytorch Omniglot class to do the downloads and data checks
            self.old_train = Omniglot(self.root, True, None, None, self.download)
            self.old_test = Omniglot(self.root, False, None, None, self.download)

            # After downloads images lie in root/omniglot-py/images_{background, evaluation}/alphabet/character/image.png
            # Retaining only those that lie in training or test set
            # TODO: look for more elegant solutions with glob and trailing slashes
            trailing_slash = "" if self.root[-1] == "/" else "/"
            image_paths = glob.glob(self.root + trailing_slash + "*/*/*/*/*.png")
            is_test_class = lambda path: any([alphabet == path.split("/")[-3] for alphabet in OMNIGLOT_TEST_CLASSES])
            if self.train:
                image_paths = [x for x in image_paths if not is_test_class(x)]
            else:
                image_paths = [x for x in image_paths if is_test_class(x)]

            # Mapping remaining characters to classes
            extract_character = lambda path: path.split("/")[-3] + "/" + path.split("/")[-2]
            characters = set([extract_character(x) for x in image_paths])
            character_mapping = dict(zip(list(characters), range(len(characters))))
            self.n_all_classes = len(self.rotations) * len(characters)

            # Reading images into memory
            self.data = torch.zeros((self.images_per_class * self.n_all_classes, 1, self.image_size, self.image_size))
            self.target = torch.zeros((self.images_per_class * self.n_all_classes,), dtype=torch.long)
            for rotation_idx, rotation in enumerate(self.rotations):
                for image_idx, image_path in enumerate(image_paths):
                    target_idx = character_mapping[extract_character(image_path)] + rotation_idx * len(characters)
                    image = Image.open(image_path, mode="r").convert("L")
                    processed_image = image.rotate(rotation).resize((self.image_size, self.image_size))

                    self.data[rotation_idx * len(image_paths) + image_idx] = ToTensor()(processed_image)
                    self.target[rotation_idx * len(image_paths) + image_idx] = target_idx

            # Recording the mapping of classes to corresponding indices
            self.target_mapping = {x: [] for x in range(self.n_all_classes)}
            for (target_idx, idx) in zip(self.target, range(self.target.shape[0])):
                self.target_mapping[int(target_idx)].append(idx)

            with open(cache_pkl, 'wb') as f:
                class_members = [self.data, self.target, self.n_all_classes, self.target_mapping]
                pickle.dump(class_members, f)

        self.shuffle_classes()

    def __len__(self):
        return self.images_per_class * self.n_classes

    def __getitem__(self, index):
        original_idx = self.data_indices_flat[index]
        image, true_class = self.data[original_idx], self.target[original_idx]
        mapped_class = self.restricted_class_mapping[int(true_class)]
        if self.train:
            # Adding gaussian noise to the image (assuming the image is already torch.tensor in [0, 1] range)
            noise = self.noise_std * torch.rand(image.shape)
            return torch.clamp(image + noise, 0, 1), mapped_class
        else:
            return image, mapped_class

    def shuffle_classes(self):
        restricted_class_set = np.random.choice(self.n_all_classes, self.n_classes, False)
        self.restricted_class_mapping = dict(zip(restricted_class_set, range(self.n_classes)))
        data_indices = [self.target_mapping[idx] for idx in restricted_class_set]
        self.data_indices_flat = [x for indices in data_indices for x in indices]
