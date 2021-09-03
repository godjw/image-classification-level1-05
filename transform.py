from torchvision import transforms as T


class BaseTransform:
    def __init__(self, resize, mean, std):
        self.transforms = [
            T.CenterCrop((360, 300)),
            T.Resize(resize, T.InterpolationMode.BICUBIC),
            
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]

    def __call__(self, image):
        return T.Compose(self.transforms)(image)


class CustomTransform(BaseTransform):
    def __init__(self, resize, mean, std):
        super().__init__(resize=resize, mean=mean, std=std)
        self.transforms = [*self.transforms, T.RandomHorizontalFlip(p=0.5)]
