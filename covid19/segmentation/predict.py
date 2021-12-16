import segmentation_models as sm
import tensorflow as tf
from PIL import Image
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from tqdm import tqdm

from covid19.segmentation.dataset import DatasetImages, DataloaderImages
from covid19.segmentation.utils import *

# Fix sm
sm.set_framework('tf.keras')
sm.framework()


def predict(pred_dataset, output_path, model_path, batch_size, use_multiprocessing, workers, save_overlay=True):
    # Build dataloader
    predict_dataloader = DataloaderImages(pred_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(filepath=model_path, compile=False)
    model.summary()

    # Compile the model
    model.compile(loss=bce_jaccard_loss, metrics=[iou_score])

    # Predicting images
    print("Predicting images...")
    for img_ids, batch in tqdm(predict_dataloader, total=len(predict_dataloader)):
        pred = model.predict(batch, use_multiprocessing=use_multiprocessing, workers=workers)

        # Save images in batch
        for img_id, img, img_pred in zip(img_ids, batch, pred):
            # From 0-1 to 0-255
            img255 = (img.squeeze()).astype(np.uint8)
            pred_img255 = ((img_pred.squeeze() > 0.5) * 255).astype(np.uint8)

            # Convert to PIL
            pil_img = Image.fromarray(img255)
            pil_img_pred = Image.fromarray(pred_img255)

            # Save images
            pil_img_pred.save(os.path.join(output_path, img_id))
            # print(f"Image save! {img_id}")

            # Save overlay
            if save_overlay:
                # Convert to RGBA
                pil_img = pil_img.convert('RGBA')
                pil_img_pred = pil_img_pred.convert('RGBA')

                # Make the background transparent
                pil_img_pred_trans = make_transparent(pil_img_pred, color=(0, 0, 0))
                pil_img_pred_trans.putalpha(75)
                overlaid_img = Image.alpha_composite(pil_img, pil_img_pred_trans)

                # Save overlaid image
                overlaid_img.save(os.path.join(output_path, "../images_masked256", img_id))
                # print(f"Overlaid image save! {img_id}")

                # Show images
                # if True:
                #     imshow(overlaid_img)
                #     plt.show()
                asd = 3


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


def get_pred_dataset(filenames, images_dir, backbone, target_size):
    # Preprocessing
    prep_fn = preprocessing_fn(custom_fn=sm.get_preprocessing(backbone))

    # Build dataset
    dataset = DatasetImages(filenames, imgs_dir=images_dir, da_fn=ts_da_fn(*target_size),
                            preprocess_fn=prep_fn, target_size=target_size, memory_map=False)
    return dataset


def get_images_with_no_mask(images_dir, masks_dir):
    # Check if all masks match an images
    img_files = [os.path.splitext(os.path.split(file)[1])[0] for file in
                 glob.glob(os.path.join(images_dir, "*.png"))]
    mask_files = [os.path.splitext(os.path.split(file)[1])[0] for file in
                  glob.glob(os.path.join(masks_dir, "*.png"))]

    # Get images with no mask
    images_with_nomask = set(img_files).difference(mask_files)
    print("Summary:")
    print(f"\t- Total images: {len(img_files)}")
    print(f"\t- Total masks: {len(mask_files)}")
    print(f"\t- Total images with no mask: {images_with_nomask}")
    print(f"\t- Total masks with no image: {len(set(mask_files).difference(img_files))}")

    # Add extension
    images_with_nomask = [f"{fname}.png" for fname in images_with_nomask]
    return images_with_nomask


def main(model_path, batch_size=32, backbone="resnet34", base_path=".",
         target_size=(256, 256), use_multiprocessing=True, workers=8):
    # Vars
    images_dir = os.path.join(base_path, "images256")
    masks_dir = os.path.join(base_path, "masks256")
    pred_masks_dir = os.path.join(base_path, "masks256_pred")

    # Get images with no mask
    images_with_nomask = get_images_with_no_mask(images_dir, masks_dir)

    # Get data
    predict_dataset = get_pred_dataset(filenames=images_with_nomask, images_dir=images_dir,
                                       backbone=backbone, target_size=target_size)

    # Predict images
    predict(predict_dataset, output_path=pred_masks_dir, model_path=model_path, batch_size=batch_size,
            use_multiprocessing=use_multiprocessing, workers=workers)


if __name__ == "__main__":
    BASE_PATH = "/home/scarrion/datasets/covid19/front"
    MODEL_PATH = os.path.join("/home/scarrion/projects/mltests/covid19/code/.outputs/models/", "unet_resnet34.h5")
    BACKBONE = "resnet34"

    # Run
    main(model_path=MODEL_PATH, backbone=BACKBONE, base_path=BASE_PATH)
    print("Done!")
