import itertools
import math
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn, Tensor
from torchvision.transforms import ToTensor

from src.dataset.GTSRBDatasetWithNegatives import GTSRBDatasetWithNegatives
from src.networks.CnnDetectorWithSlidingWindow import CnnDetectorWithSlidingWindow


class FrameAnalyzer:

    def __init__(self, detector: CnnDetectorWithSlidingWindow, device: str, train: bool,
                 width: int, height: int,
                 step_size: int = 3, iou_threshold: float = 0.2):
        detector.eval()
        detector = detector.to(device)
        self.train = train
        self.detector_model = detector
        self.device = device
        self.step_size: int = step_size
        self.tensor_transformation = ToTensor()
        self.sm = nn.Softmax(dim=1)
        self.patch_w: int = detector.IMG_SIZE
        self.patch_h: int = self.patch_w
        self.resize_sizes = self.get_sizes(width, height, patch_size=self.patch_w, patch_stride=step_size)
        self.out_images = {}
        self.iou_threshold = iou_threshold

    @staticmethod
    def get_sizes(original_w: int, original_h: int,
                  patch_size: int, patch_stride: int,
                  min_w: int = 100, min_h: int = 100) -> list:
        n_res = 5  # 10
        delta_w = int((original_w - min_w) / n_res)
        w_range = list(range(min_w, int(original_w), delta_w))
        # w_range = list(range(min_w, int(original_w / 1.5), delta_w))
        ratio_range = [w / original_w for w in w_range]
        sizes = [(int(original_w * r), int(original_h * r), r) for r in ratio_range]
        sizes = [(x, y, r, FrameAnalyzer.get_grid(x, y, patch_size, patch_stride)) for x, y, r in sizes]
        return sizes

    @staticmethod
    def get_grid(img_width: int, img_height: int, patch_size: int, patch_stride: int) -> list:
        n = math.ceil((img_width - patch_size) / patch_stride)  # img.size[1]
        m = math.ceil((img_height - patch_size) / patch_stride)  # img.size[0]
        # [x1,x2,...,xn] m times
        x_ = list(range(0, img_width - patch_size, patch_stride)) * m
        # [y1 (n times),y2 (n times), ..., ym (n times)]
        y_ = [[v] * n for v in list(range(0, img_height - patch_size, patch_stride))]
        y_ = list(itertools.chain(*y_))
        return [(x, y) for x, y in zip(x_, y_)]

    def video_analyzer(self, stream) -> None:

        cap = cv2.VideoCapture(stream)

        _, img = cap.read()
        self.resize_sizes = self.get_sizes(img.shape[1], img.shape[0])
        print(self.resize_sizes)

        self.out_images = {}

        while True:
            _, img = cap.read()

            # cv2.imshow('img', img)

            rectangles = self.get_detected_rectangles(img)
            print(len(rectangles))

            if not self.train:
                for rect in rectangles:
                    x1, y1, x2, y2 = rect
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

            cv2.imshow('img', img)
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(self.out_images.keys()) and self.train:
            # save to file and add to json
            folder_path = os.path.dirname(__file__) + "/dataset/GTSRB/Negative/images/"
            print("Storing", len(self.out_images.keys()), "negative examples")
            for out_filename in self.out_images.keys():
                img: Image = self.out_images[out_filename]
                path = folder_path + out_filename
                img.save(path)
                # print(path)
            GTSRBDatasetWithNegatives.add_images_to_negative_examples_json(list(self.out_images.keys()),
                                                                           overwrite=False)  # TODO false

    def sliding_window_on_scaled_img(self, image: Image, size: tuple):
        if image is None:
            return []

        w, h, scale_factor, grid = size
        scale_factor = 1 / scale_factor
        resized_image = cv2.resize(image, (w, h))

        window_img: Image = Image.fromarray(resized_image)

        labels, scores = self.detector_model.run_sliding_window(window_img)

        out = []
        if len(labels) == len(grid):  # TODO CHECK
            for label, score, position in zip(labels, scores, grid):
                if label == GTSRBDatasetWithNegatives.STREET_SIGN_LABEL_IDX:
                    # positions = np.array(grid)[np.array(labels) == GTSRBDatasetWithNegatives.STREET_SIGN_LABEL_IDX]
                    x, y = position
                    out.append((
                        int(scale_factor * x), int(scale_factor * y),  # x1, y1
                        int(scale_factor * (x + self.patch_w)), int(scale_factor * (y + self.patch_h)),  # x2,y2
                        score
                    ))

        # TODO CHECK
        #  positions = np.array(grid)[np.array(labels) == GTSRBDatasetWithNegatives.STREET_SIGN_LABEL_IDX]
        #  IndexError: boolean index did not match indexed array along dimension 0; dimension is 234 but corresponding boolean dimension is 247

        return out

    def get_detected_rectangles(self, img: Image) -> list:

        overlay_rectangles = []

        for size in self.resize_sizes:
            for rect in self.sliding_window_on_scaled_img(img, size):
                overlay_rectangles.append(rect)

        # non maximum suppression https://pytorch.org/docs/stable/torchvision/ops.html
        if len(overlay_rectangles):
            overlay_rectangles_np = np.array(overlay_rectangles)
            t = Tensor(overlay_rectangles_np[:, 0:4])  # isolate x1,y1,x2,y2
            scores = Tensor(overlay_rectangles_np[:, 4]) # isolate score
            nms_rects = torch.ops.torchvision.nms(t, scores, self.iou_threshold).tolist()
            return overlay_rectangles[nms_rects[:50], :]

        return []
