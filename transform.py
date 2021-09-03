from torchvision import transforms as T


class BaseTransform:
    """
    Base tranform class that apllies to validation dataset.
    It contains [Resize, ToTensor, Normalize] by default.

    Args:
        resize (sequence): Size that an image is resized to
        mean (sequence): Sequence of means for each channel
        std (sequence): Sequence of standard deviations for each channel
    """

    def __init__(self, resize, mean, std):
        self.transforms = [
            T.Resize(resize, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]

    def __call__(self, image):
        return T.Compose(self.transforms)(image)


class CustomTransform(BaseTransform):
    """
    Custom tranform class that apllies to train dataset.
    It contains [BaseTransform, RandomHorizontalFlip(p=0.5)] by default.

    Args:
        resize (sequence): Size that an image is resized to
        mean (sequence): Sequence of means for each channel
        std (sequence): Sequence of standard deviations for each channel
    """

    def __init__(self, resize, mean, std):
        super().__init__(resize=resize, mean=mean, std=std)
        self.transforms = [*self.transforms, T.RandomHorizontalFlip(p=0.5)]
