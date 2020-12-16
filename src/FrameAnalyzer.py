import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import ToTensor

from src.custom_transformations.CustomToGrayScale import CustomToGrayScale
from src.dataset.GTSRBDataset import GTSRBDataset
from src.networks.CnnClassifier import CnnClassifier
from src.networks.SlidingWindow import SlidingWindow


class FrameAnalyzer:

    def __init__(self, device: str,
                 width: int, height: int, stride: int = 5, iou_threshold: float = 0.5,
                 classifier: CnnClassifier = None):
        self.device = device
        self.classifier = classifier
        if classifier is None:
            self.classifier = CnnClassifier(device).to(device)
            self.classifier.load_from_file()
        #
        self.patch_w: int = CnnClassifier.PATCH_SIZE
        self.patch_h: int = self.patch_w
        self.stride: int = stride
        #
        # print("Frame analyzer has full res size w={}, h={} and stride {}".format(width, height, stride))
        self.resize_sizes = SlidingWindow.get_sizes(width, height, patch_size=self.patch_w, patch_stride=stride)
        # print(len(self.resize_sizes), "scaling")
        # print([(x, y) for x, y, r, s in self.resize_sizes])
        #
        self.out_images = {}
        self.iou_threshold = iou_threshold
        self.sw = SlidingWindow(self.classifier, self.device, CnnClassifier.PATCH_SIZE, self.stride)

    def scale_img_run_sliding_window(self, image: Image, size: tuple) -> list:
        if image is None:
            return []

        w, h, scale_factor, grid = size
        scale_factor = 1 / scale_factor

        # rescale image
        resized_image = image.resize((w, h))

        # image to tensor
        img_tensor: Tensor = (ToTensor())(resized_image).float()
        img_tensor.requires_grad = False

        # unfold patches and run classification
        labels, scores = self.sw.unfold_and_classify(img_tensor)

        out = []

        # compute rectangles of regions classified as street signs
        for label_idx, score, position in zip(labels, scores, grid):
            if label_idx != GTSRBDataset.NO_STREET_SIGN_LABEL_IDX:
                x, y = position
                out.append((
                    int(scale_factor * x), int(scale_factor * y),  # x1, y1
                    int(scale_factor * (x + self.patch_w)), int(scale_factor * (y + self.patch_h)),  # x2, y2
                    label_idx,  # index of the label
                    score  # score
                ))

        return out

    def get_detected_street_signs(self, img: Image, limit: int = 4) -> list:

        overlay_rectangles = []
        gray_img = (CustomToGrayScale())(img)
        for size in self.resize_sizes:
            overlay_rectangles += self.scale_img_run_sliding_window(gray_img, size)

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
