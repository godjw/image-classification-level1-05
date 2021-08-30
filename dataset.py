# System Libs.
import random
from pathlib import Path

# Other Libs
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class TrainInfo():
    def __init__(self, file_dir=None, data_dir='/opt/ml/input/data/train/images'):
        # Train info data
        self.data = pd.read_csv(file_dir) if file_dir else pd.read_csv(
            'processed_train.csv')

        # Update directory
        self.data_dir = Path(data_dir)
        self.update_data_dir()

    def update_data_dir(self):
        paths = self.data['FullPath']
        paths_pre = paths.copy()
        paths_pre.loc[:] = str(self.data_dir)
        paths_post = paths.str.split('/images').str[1]
        self.data['FullPath'] = paths_pre.str.cat(paths_post)

    def split_dataset(self, crit_col='path', val_size=0.2, shuffle=True, random_state=32):
        if random_state:
            random.seed(random_state)

        _idxs = set(self.data[crit_col].unique())
        _size = len(_idxs)
        _size_valid = int(_size * val_size)

        # Split DataFrame
        valid_idxs = set(random.sample(_idxs, _size_valid))
        valid_df = self.data.loc[self.data[crit_col].isin(valid_idxs)]

        train_idxs = _idxs - valid_idxs
        train_df = self.data.loc[self.data[crit_col].isin(train_idxs)]

        # Split Result
        split_result = dict(origin=self.data, train=train_df, valid=valid_df)
        split_result = self._split_result(split_result)

        return train_df, valid_df, split_result

    def _split_result(self, df_dict, col_list=['Mask', 'Age', 'Gender']):
        dist_list = []
        for name, df in df_dict.items():
            dist_df_list = []
            for col in col_list:
                # Dist. info
                dist_count = pd.DataFrame(df[col].value_counts())
                dist_count.columns = ['Count']
                dist_ratio = pd.DataFrame(df[col].value_counts(True))
                dist_ratio.columns = ['Ratio']

                # Construct & append dataframe
                _dist_df = pd.concat([dist_count, dist_ratio], axis=1)
                _dist_df.index = pd.MultiIndex.from_product(
                    [[col], _dist_df.index])
                dist_df_list.append(_dist_df)
            dist_df = pd.concat(dist_df_list, axis=0)
            dist_df.columns = pd.MultiIndex.from_product(
                [[name], dist_df.columns])
            dist_list.append(dist_df)
        dist_info = pd.concat(dist_list, axis=1)

        return dist_info


class MaskBaseDataset(Dataset):

    def __init__(self, data_dir, mean=None, std=None, train_dir=None):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std

        self.train_info = TrainInfo(
            train_dir=train_dir, data_dir=self.data_dir)
        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        train_df = self.train_info.data
        self.image_paths = list(train_df['FullPath'])
        self.mask_labels = list(train_df['Mask'])
        self.gender_labels = list(train_df['Gender'])
        self.age_labels = list(train_df['Age'])

    def calc_statistics(self):
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
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(
            mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index):
        return self.mask_labels[index].value

    def get_gender_label(self, index):
        return self.gender_labels[index].value

    def get_age_label(self, index):
        return self.age_labels[index].value

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return (mask_label, gender_label, age_label)

    @staticmethod
    def denormalize_image(image, mean, std):
        _img = image.copy()
        _img *= std
        _img += mean
        _img *= 255.0
        _img = np.clip(_img, 0, 255).astype(np.uint8)
        return _img

    def split_dataset(self, val_size) -> Tuple[Subset, Subset]:
        n_val = int(len(self) * val_size)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return (train_set, val_set)


class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = BaseTransform(
            resize=resize,
            mean=mean,
            std=std
        )

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
