# System Libs.
import random
from pathlib import Path

# Other Libs
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

from transform import BaseTransform


class TrainInfo:
    """ 
    Class for manage/manipulate the dataframe for PyTorch Dataset construction.
    The dataframe should contain 1) Paths about the image files, 2) Labels for matching images.
    
    Args:
        file_dir (str or pathlib.Path, optional):  Dataframe csv file path. Defaults to None.
        data_dir (str or pathlib.Path, optional):  Parent path for image files. Defaults to "/opt/ml/input/data/train/images"
        new_dataset (bool, optional):  Whether the data_dir needs to be updated. Defaults to False.
    """

    def __init__(
        self, file_dir=None, data_dir="/opt/ml/input/data/train/images", new_dataset=False,
    ):
        self.data = pd.read_csv(file_dir) if file_dir else pd.read_csv("metadata/processed_train.csv")
        self.data_dir = Path(data_dir)

        if new_dataset == False:
            self.update_data_dir()

    def update_data_dir(self):
        """Update path data for image files.
        """
        paths = self.data["FullPath"]
        paths_pre = paths.copy()
        paths_pre.loc[:] = str(self.data_dir)
        paths_post = paths.str.split("/images").str[1]
        self.data["FullPath"] = paths_pre.str.cat(paths_post)

    def split_dataset(self, val_size=0.2, crit_col="path", shuffle=True, random_state=32):
        """
        Split the data info to train info and validation info.

        Args:
            val_size (float, optional): Ratio for validation set. Defaults to 0.2.
            crit_col (str, optional): Split by column. Defaults to "path".
            shuffle (bool, optional): Shuffle. Defaults to True.
            random_state (int, optional): Random seed number. Defaults to 32.

        Returns:
            train_df (pd.DataFrame): Train set info
            valid_df (pd.DataFrame): Validation set info
            split_result (pd.DataFrame): Split result(Distribution info for each feature)
        """
        if random_state:
            random.seed(random_state)
        _idxs = set(self.data[crit_col].unique())
        _size = len(_idxs)
        _size_valid = int(_size * val_size)

        valid_idxs = set(random.sample(_idxs, _size_valid))
        valid_df = self.data.loc[self.data[crit_col].isin(valid_idxs)]

        train_idxs = _idxs - valid_idxs
        train_df = self.data.loc[self.data[crit_col].isin(train_idxs)]
        # age offset
        train_df = train_df.query("(age <= 25) | (age >=30 & age <= 58) | (age >= 60)")

        split_result = dict(origin=self.data, train=train_df, valid=valid_df)
        split_result = self._split_result(split_result)

        return train_df, valid_df, split_result

    def _split_result(self, df_dict, col_list=["Mask", "Age", "Gender"]):
        dist_list = []
        for name, df in df_dict.items():
            dist_df_list = []
            for col in col_list:
                # Dist. info
                dist_count = pd.DataFrame(df[col].value_counts())
                dist_count.columns = ["Count"]
                dist_ratio = pd.DataFrame(df[col].value_counts(True))
                dist_ratio.columns = ["Ratio"]

                # Construct & append dataframe
                _dist_df = pd.concat([dist_count, dist_ratio], axis=1)
                _dist_df.index = pd.MultiIndex.from_product([[col], _dist_df.index])
                dist_df_list.append(_dist_df)
            dist_df = pd.concat(dist_df_list, axis=0)
            dist_df.columns = pd.MultiIndex.from_product([[name], dist_df.columns])
            dist_list.append(dist_df)
        dist_info = pd.concat(dist_list, axis=1)

        return dist_info


class MaskBaseDataset(Dataset):
    """
    Generate Dataset from Info dataframe.

    Args:
        data_info (pd.DataFrame, optional): Info dataset for dataset construction.
        mean (sequence, optional): Mean info for normalize.
        std (sequence, optional): Std info for normalize.
        path_col (str, optional): Path info for reading image files.
        label_col (str, optional): Label info.
    """

    def __init__(self, data_info, mean=None, std=None, path_col="FullPath", label_col="Class"):
        """Initialize
        """
        self.data_info = data_info
        self.path_col = path_col
        self.path_label = label_col

        self.mean = mean
        self.std = std
        self.transform = None
        self.num_classes = 18
        if label_col == "Class Mask" or "Class Age":
            self.num_classes = 3
        elif label_col == "Class Gender":
            self.num_classes = 2
        self.setup()
        self.calc_statistics()

    def setup(self):
        """Setup path, label, class info.
        """
        self.img_paths = list(self.data_info[self.path_col])
        self.labels = list(self.data_info[self.path_label])
        self.num_classes = len(set(self.labels))

    def calc_statistics(self):
        """Calculate and update mean & std info.
        """
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("Calculating statistics... This might take a while")
            sums = []
            squared = []
            for img_path in tqdm(self.img_paths):
                image = np.array(Image.open(img_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        """
        Apply transform.

        Args:
            transform (torch.transforms.compose): Transforms.
        """
        self.transform = transform

    def __getitem__(self, index):
        """
        Get item module

        Args:
            index (int): Index

        Returns:
            image_transform (torch.tensor): Transformed image.
            label (int): Label
        """
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        label = int(self.get_label(index))

        image_transform = self.transform(image)
        return image_transform, label

    def __len__(self):
        """
        Return dataset length.

        Returns:
            length (int): Length of the dataset.
        """
        return len(self.img_paths)

    def read_image(self, index):
        """
        Read and return the image corresponding to the index.

        Args:
            index (int): Index

        Returns:
            image (Image): Image
        """
        img_path = self.img_paths[index]
        return Image.open(img_path)

    def get_label(self, index):
        """
        Read and return the label corresponding to the index.

        Args:
            index (int): index

        Returns:
            label (str or int): label
        """
        return self.labels[index]

    @staticmethod
    def decode_multi_class(multi_class_label):
        """
        Decompose combined label to mask label, gender label, and age label.

        Args:
            multi_class_label (int): label

        Returns:
            label_list (tuple): mask label, gender label, and age label.
        """
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return (mask_label, gender_label, age_label)

    @staticmethod
    def denormalize_image(image, mean, std):
        """
        Denormazlie image.

        Args:
            image (Image): Image
            mean (sequence): Mean
            std (sequence): Std

        Returns:
            image (Image): Denormalized image
        """
        _img = image.copy()
        _img *= std
        _img += mean
        _img *= 255.0
        _img = np.clip(_img, 0, 255).astype(np.uint8)
        return _img


class TestDataset(Dataset):
    """
    Dataset for test data(eval).

    Args:
        img_paths (Sequence): Sequence of imeage paths.
        resize (tuple): Size for resize.
        mean (tuple, optional): Mean value. Defaults to (0.548, 0.504, 0.479).
        std (tuple, optional): Std value. Defaults to (0.237, 0.247, 0.246).
    """

    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        """Initialize.
        """
        self.img_paths = img_paths
        self.transform = BaseTransform(resize=resize, mean=mean, std=std)

    def __getitem__(self, index):
        """Get item.
        """
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        """Length of dataset.
        """
        return len(self.img_paths)
