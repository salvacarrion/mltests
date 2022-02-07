import segmentation_models as sm
from covid19v2.segmentation.da import preprocessing_fn

def force_2d(img):
    if len(img.shape) == 2:
        return img
    elif len(img.shape) == 3:
        return img[:, :, 0]
    else:
        raise ValueError("Images must have either 2D or 3D")


def get_model(backbone):
    if backbone == "resnet34":
        from segmentation_models import Unet
        model = Unet("resnet34", encoder_weights='imagenet', classes=1, activation='sigmoid', encoder_freeze=True)
        prep_fn = preprocessing_fn(custom_fn=sm.get_preprocessing("resnet34"))
    else:
        raise ValueError("Unknown backbone")
    return model, prep_fn


def make_transparent(img, color=(0, 0, 0)):
    rgba = img.convert("RGBA")
    datas = rgba.getdata()

    newData = []
    for item in datas:
        if item[0] == color[0] and item[1] == color[1] and item[2] == color[2]:  # finding black colour by its RGB value
            # storing a transparent value when we find a black colour
            newData.append((0, 0, 0, 0))
        else:
            newData.append(item)  # other colours remain unchanged

    rgba.putdata(newData)
    return rgba
