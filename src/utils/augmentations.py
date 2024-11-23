from torchvision import transforms
from typing import Union, Tuple, Optional
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def make_transforms(
        resize: Union[None, int, tuple[int]] = None,
        resize_interpolation: int = 3,
        center_crop: Union[None, int] = None,
        rand_size: Union[None, int, tuple[int]] = None,
        rand_size_interpolation: int = 3,
        rand_size_scale: Union[None, tuple[float]] = (0.08, 1.0),
        rand_size_ratio: Union[None, tuple[float]] = (3.0 / 4.0, 4.0 / 3.0),
        rand_horizontal_flip=0.0,
        use_timm: bool = False,
        timm_input_size: Union[int, Tuple[int, int], Tuple[int, int, int]] = 256,
        timm_is_training: bool = False,
        timm_color_jitter: Union[float, Tuple[float, ...]] = 0.4,
        timm_color_jitter_prob: Optional[float] = None,
        timm_auto_augment: Optional[str] = None,
        timm_interpolation: str = 'bicubic',
        timm_re_prob: float = 0.,
        timm_re_mode: str = 'const',
        timm_re_count: int = 1,
        **kwargs
):
    if use_timm:
        transform = create_transform(
            input_size=timm_input_size,
            is_training=timm_is_training,
            color_jitter=timm_color_jitter,
            color_jitter_prob=timm_color_jitter_prob,
            auto_augment=timm_auto_augment,
            interpolation=timm_interpolation,
            re_prob=timm_re_prob,
            re_mode=timm_re_mode,
            re_count=timm_re_count,
        )
        return transform

    transform_list = []

    if resize is not None:
        transform_list += [transforms.Resize(size=resize, interpolation=resize_interpolation)]
    elif rand_size is not None:
        transform_list += [transforms.RandomResizedCrop(size=rand_size,
                                                        scale=rand_size_scale,
                                                        ratio=rand_size_ratio,
                                                        interpolation=rand_size_interpolation)]
    if center_crop is not None:
        transform_list += [transforms.CenterCrop(size=center_crop)]
    if rand_horizontal_flip > 0:
        transform_list += [transforms.RandomHorizontalFlip(p=rand_horizontal_flip)]

    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]

    transform = transforms.Compose(transform_list)
    return transform
