import itertools
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import ToTensor

from src.dataset.GTSRBDatasetWithNegatives import GTSRBDatasetWithNegatives
from src.networks.CnnClassifier import CnnClassifier
from src.networks.CnnDetector import CnnDetector
from src.networks.CnnDetectorWithSlidingWindow import CnnDetectorWithSlidingWindow


class FrameAnalyzer:

    def __init__(self, device: str, train: bool,  # classifier: CnnClassifier,
                 width: int, height: int, stride: int = 5, score_threshold: float = 0.5, iou_threshold: float = 0.5):
        self.device = device
        self.classifier = CnnClassifier(device).to(device)
        self.classifier.load_from_file()
        # self.detector = CnnDetector(device).to(device)
        # self.detector.load_from_file()
        #
        self.train = train
        self.score_threshold = score_threshold
        self.patch_w: int = CnnDetector.PATCH_SIZE
        self.patch_h: int = self.patch_w
        self.stride: int = stride
        #
        # print("Frame analyzer has full res size w={}, h={} and stride {}".format(width, height, stride))
        self.resize_sizes = self.get_sizes(width, height, patch_size=self.patch_w, patch_stride=stride)
        # print(len(self.resize_sizes), "scaling")
        # print([(x, y) for x, y, r, s in self.resize_sizes])
        #
        self.out_images = {}
        self.iou_threshold = iou_threshold
        # self.sw = CnnDetectorWithSlidingWindow(self.detector, self.device, CnnDetector.PATCH_SIZE, self.stride)
        self.sw = CnnDetectorWithSlidingWindow(self.classifier, self.device, CnnDetector.PATCH_SIZE, self.stride)

    @staticmethod
    def get_sizes(original_w: int, original_h: int,
                  patch_size: int, patch_stride: int,
                  min_w: int = 100, min_h: int = 100, n_res=7) -> list:

        max_h = original_h / 3

        # delta_w = int((original_w - min_w) / n_res)
        # w_range = list(range(min_w, int(original_w), delta_w))

        rng = [min_h + (max_h - min_h) / n_res * r for r in range(0, n_res, 1)]
        # rng = list(range(min_w, int(original_w / 4), delta_w))  # TODO change here

        ratio_range = [h / original_h for h in rng]

        sizes = [(int(original_w * r), int(original_h * r), r) for r in ratio_range]
        sizes = [(x, y, r, FrameAnalyzer.get_grid(x, y, patch_size, patch_stride)) for x, y, r in sizes]
        return sizes

    @staticmethod
    def get_grid(img_width: int, img_height: int, patch_size: int, patch_stride: int) -> list:
        # [x1,x2,...,xn] m times
        x_ = list(range(0, img_width + 1 - patch_size, patch_stride))
        # [y1 (n times),y2 (n times), ..., ym (n times)]
        y_ = list(range(0, img_height + 1 - patch_size, patch_stride))
        m, n = len(y_), len(x_)
        x_ = x_ * m
        y_ = [[v] * n for v in y_]
        y_ = list(itertools.chain(*y_))
        return [(x, y) for x, y in zip(x_, y_)]

    def sliding_window_on_scaled_img(self, image: Image, size: tuple) -> list:
        if image is None:
            return []

        w, h, scale_factor, grid = size
        scale_factor = 1 / scale_factor
        resized_image = image.resize((w, h))

        img_tensor: Tensor = (ToTensor())(resized_image).float()
        img_tensor.requires_grad = False

        labels, scores = self.sw.unfold_and_classify(img_tensor)

        out = []

        for label_idx, score, position in zip(labels, scores, grid):
            if score > self.score_threshold:
                x, y = position
                out.append((
                    int(scale_factor * x), int(scale_factor * y),  # x1, y1
                    int(scale_factor * (x + self.patch_w)), int(scale_factor * (y + self.patch_h)),  # x2,y2
                    label_idx,
                    score
                ))

        return out

    def get_detected_rectangles(self, img: Image, limit: int = 4) -> list:

        overlay_rectangles = []
        for size in self.resize_sizes:
            overlay_rectangles += self.sliding_window_on_scaled_img(img, size)

        out = []

        # non maximum suppression https://pytorch.org/docs/stable/torchvision/ops.html
        if len(overlay_rectangles):
            overlay_rectangles_np = np.array(overlay_rectangles)
            t = Tensor(overlay_rectangles_np[:, 0:4]).detach()  # isolate x1,y1,x2,y2
            scores = Tensor(overlay_rectangles_np[:, 4]).detach()  # isolate score
            nms_rects = torch.ops.torchvision.nms(t, scores, self.iou_threshold).tolist()
            print("Before NMS", len(overlay_rectangles_np), "After NMS", len(nms_rects))
            out = overlay_rectangles_np[nms_rects[:limit], :]

            del t
            del scores
            del nms_rects

        return out
