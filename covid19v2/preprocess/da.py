import albumentations as A
import cv2


def _da_negative(image, **kwargs):
    return 255 - image


# define heavy augmentations
def tr_da_fn(height, width):
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.10, rotate_limit=7, shift_limit=0.10, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Perspective(scale=(0.025, 0.04), p=0.3),
        A.RandomResizedCrop(height=height, width=width, scale=(0.9, 1.0), p=0.3),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
                A.RandomContrast(limit=0.2, p=1.0),
            ],
            p=0.5,
        ),

        A.OneOf(
            [
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                A.Blur(blur_limit=[2, 3], p=1.0),
                A.GaussNoise(var_limit=(5, 25), p=1.0),
                # A.MotionBlur(blur_limit=3, p=1.0),
            ],
            p=0.5,
        ),

        A.Lambda(image=_da_negative, p=0.2),

        A.LongestMaxSize(max_size=max(height, width), always_apply=True),
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
    ]
    return A.Compose(train_transform)


def ts_da_fn(height, width):
    _transform = [
        A.LongestMaxSize(max_size=max(height, width), always_apply=True),
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
    ]
    return A.Compose(_transform)


def preprocessing_fn(custom_fn):
    _transform = [
        A.Lambda(image=custom_fn),
    ]
    return A.Compose(_transform)


def da_resize_pad_fn(height, width):
    da_transform = []
    da_transform += [
        A.LongestMaxSize(max_size=max(height, width), interpolation=cv2.INTER_LANCZOS4, always_apply=True),
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
        A.Resize(height=height, width=width),  # Workaround for the rounding bug
    ]
    return A.Compose(da_transform)
