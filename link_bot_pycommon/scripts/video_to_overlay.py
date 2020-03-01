import argparse
import pathlib

import cv2

from PIL import ImageColor, Image
import numpy as np

from link_bot_pycommon.args import my_formatter


def main():
    parser = argparse.ArgumentParser("video -> combined/overlaid picture with alpha", formatter_class=my_formatter)
    parser.add_argument('video', type=pathlib.Path)
    parser.add_argument('--start-frame', type=int, help='start frame number', default=0)
    parser.add_argument('--stop-frame', type=int, help='stop frame number', default=1e3)
    parser.add_argument('--subtract-color', type=str, help='html color code, without the octothorp')

    args = parser.parse_args()

    video_path = args.video.expanduser()
    if not video_path.exists():
        print("{} does not exist".format(args.video))

    video = cv2.VideoCapture(str(video_path))

    overlay_image = None
    frame_idx = 0
    cv2.namedWindow('')
    while True:
        success, image = video.read()

        frame_idx += 1

        if not success:
            break
        if frame_idx < args.start_frame:
            continue
        if frame_idx >= args.stop_frame:
            break
        if args.subtract_color:
            # for all pixels equal to subtract color, set the alpha channel to zero.
            rgb = ImageColor.getrgb("#{}".format(args.subtract_color))
            background_mask = np.all(image == rgb, axis=2)
            background_indeces = np.argwhere(background_mask)
            background_rows = background_indeces[:, 0]
            background_cols = background_indeces[:, 1]
            alpha = np.ones([image.shape[0], image.shape[1], 1], dtype=image.dtype) * 0.999
            alpha[background_rows, background_cols] = 0
            image[background_rows, background_cols] = 0
            image = np.concatenate((image, alpha), axis=2)

        if overlay_image is None:
            overlay_image = image
        else:
            overlay_image += image

        cv2.imshow("", overlay_image / np.max(overlay_image))
        cv2.waitKey(1)

    overlay_image = overlay_image[:, :, :3]
    normalized_overlay_image = overlay_image / np.max(overlay_image) * 255
    cv2.imwrite("overlay.png", normalized_overlay_image)



if __name__ == '__main__':
    main()
