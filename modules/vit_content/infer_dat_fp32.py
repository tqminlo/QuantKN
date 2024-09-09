import numpy as np
from keras.layers import *
import keras
from keras.models import Model
import cv2
from models.dat_model import DAT


# model = keras.models.load_model("saved/dat.h5")
# model = keras.models.load_model("saved/dat_fix_gelu.h5")
# model = keras.models.load_model("saved/dat_fix.h5")
# model = keras.models.load_model("saved/dat_ln_fused.h5")
# model.save_weights("saved/dat_ln_fused_w.h5")
model = DAT()()
model.load_weights("saved/dat_v1.3_w.h5")
# model.summary()
# model = keras.models.load_model("saved/dat_v1.2.h5")
# model = keras.models.load_model("saved/dat_v1.3.h5")


def load_image(filepath):
    image = cv2.imread(filepath)  # H, W, C
    orig_shape = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = cv2.resize(image, (518, 518))
    image = np.expand_dims(image, axis=0)

    return image, orig_shape


def show_depth(depth, inp, org_shape):
    (org_w, org_h) = org_shape
    depth = cv2.resize(depth, (org_w, org_h))
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

    margin_width = 50
    caption_height = 60
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    split_region = np.ones((org_h, margin_width, 3), dtype=np.uint8) * 255
    combined_results = cv2.hconcat([inp, split_region, depth_color])

    caption_space = (
            np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8)
            * 255
    )
    captions = ["Raw image", "Depth Anything"]
    segment_width = org_w + margin_width
    for i, caption in enumerate(captions):
        # Calculate text size
        text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

        # Calculate x-coordinate to center the text
        text_x = int((segment_width * i) + (org_w - text_size[0]) / 2)

        # Add text caption
        cv2.putText(
            caption_space,
            caption,
            (text_x, 40),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )

    final_result = cv2.vconcat([caption_space, combined_results])
    # cv2.imwrite("test/relu.jpg", final_result)

    cv2.imshow("depth", final_result)
    cv2.waitKey(0)


def infer(img_path):
    image = cv2.imread(img_path)  # H, W, C
    org_h, org_w, _ = image.shape
    inp = cv2.resize(image, (518, 518))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    inp = inp / 255.
    inp = np.expand_dims(inp, axis=0)
    inp1 = np.load("saved/inp1.npy")
    inp2 = np.load("saved/inp2.npy")
    inp = [inp, inp1, inp2]

    depth = model.predict(inp)
    depth = depth[0]
    # print(depth)
    # print(depth.shape)

    show_depth(depth, image, (org_w, org_h))


if __name__ == "__main__":
    # infer("test/demo1.png")
    # infer("datasets/calib/ADE_train_00001034.jpg")
    # infer("datasets/calib/ADE_train_00009499.jpg")
    infer("datasets/calib/ADE_train_00001394.jpg")