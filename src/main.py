import argparse
import os
import random
import sys

import cv2
import numpy
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import ToTensor, ToPILImage

from src.BootStrapper import BootStrapper
from src.FrameAnalyzer import FrameAnalyzer
from src.custom_transformations.NormalizeHist import NormalizeHist
from src.custom_transformations.Rgb2Hsl import Rgb2Hsl
from src.dataset.GTSRBDataset import GTSRBDataset
from src.dataset.GTSRBDatasetWithNegatives import GTSRBDatasetWithNegatives
from src.networks.CnnClassifier import CnnClassifier
from src.networks.CnnDetector import CnnDetector

device = "cuda:0" if torch.cuda.is_available() else "cpu"

seed = 852144

if 'cuda' in device:
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)


def train_classifier(load: bool = False):
    cnn_classifier_model = CnnClassifier(device)
    if load:
        cnn_classifier_model.load_from_file()
    cnn_classifier_model.train_model()
    cnn_classifier_model.store_to_file()


def train_detector(load: bool = False):
    cnn_detector_model = CnnDetector(device)
    if load:
        cnn_detector_model.load_from_file()
    cnn_detector_model.train_model(epochs=15)
    cnn_detector_model.store_to_file()


def bootstrap():
    cnn_detector_model = CnnDetector(device)
    if not cnn_detector_model.load_from_file():
        # structure has changed
        BootStrapper.populate_json_few()
        cnn_detector_model.train_model(epochs=1)
    for iteration in range(0, 20):
        BootStrapper.add_wrongly_classified_to_negative_json(cnn_detector_model)
        cnn_detector_model.train_model(epochs=5)
        cnn_detector_model.store_to_file()


def train_detector_bootstrap(iterations: int = 10, files_per_iteration: int = 50) -> None:
    """
    The feed must have anything but street signs
    Save wrongly classified patches
    :return:
    """

    detector = CnnDetector(device)
    detector.load_from_file()

    full_res_folder = os.path.dirname(__file__) + '/dataset/GTSRB/Negative/bootstrap_full_res/'
    output_folder = os.path.dirname(__file__) + '/dataset/GTSRB/Negative/images/'
    print("Searching for new pictures in", full_res_folder, "that are classified as street signs")

    jpg_file_names = []
    for root, dirs, files in os.walk(full_res_folder):
        for name in files:
            if name.endswith(".jpg"):
                jpg_file_names.append((root, name))

    for i in range(1, iterations):
        random.shuffle(jpg_file_names)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>> Iteration {}".format(i))
        dirty = False
        new_files = []
        for root, jpg_file_name in jpg_file_names[:files_per_iteration]:
            print("\tLoading", jpg_file_name)
            pil_img = Image.open(root + jpg_file_name)
            fa = FrameAnalyzer(device, train=False, width=pil_img.size[0], height=pil_img.size[1])  # device
            if pil_img is not None:
                rectangles = fa.get_detected_rectangles(pil_img, limit=50)
                print("\t\t{} patches classified as street signs".format(len(rectangles)))
                if len(rectangles):
                    dirty = True
                for rect in rectangles:
                    x1, y1, x2, y2, label_idx, score = rect
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    fn = '{}_{}_'.format(x1, y1) + jpg_file_name
                    pil_img.crop((x1, y1, x2, y2)).save(output_folder + fn)  # crop and save
                    new_files.append(fn)
        if dirty:
            print("Adding", len(new_files), "negative examples")
            GTSRBDatasetWithNegatives.add_images_to_negative_examples_json(new_files, overwrite=False)  # TODO false
            print("\tTraining")
            detector.train_model(epochs=1)
            detector.store_to_file()


def eval_stream(stream, font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 255, 255)) -> None:
    classifier = CnnClassifier(device)
    classifier.load_from_file()

    cap = cv2.VideoCapture(stream)
    _, cv2_img = cap.read()  # fetch first frame
    fa = FrameAnalyzer(device, train=False, width=cv2_img.shape[1], height=cv2_img.shape[0])

    while True:
        _, cv2_img = cap.read()

        if cv2_img is not None:

            pil_img = Image.fromarray(cv2_img)

            rectangles = fa.get_detected_rectangles(pil_img)
            print(len(rectangles))

            for rect in rectangles:
                x1, y1, x2, y2, label_idx, score = rect
                labelName = GTSRBDataset.get_label_names()[int(label_idx)]
                cv2_img = cv2.rectangle(cv2_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                cv2_img = cv2.putText(cv2_img, labelName, (int(x1), int(y1) + 8), font, fontScale, color, 1, cv2.LINE_AA)
                cv2_img = cv2.putText(cv2_img, "{:.3f}".format(score), (int(x1), int(y1) + 16), font, fontScale, color,
                                      1, cv2.LINE_AA)

            cv2.imshow('img', cv2_img)
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


def eval_img(filepath: str, font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 255, 255)):
    classifier = CnnClassifier(device)
    classifier.load_from_file()
    pil_img = Image.open(filepath)
    fa = FrameAnalyzer(device, train=False, width=pil_img.size[0], height=pil_img.size[1])  # device

    if pil_img is not None:

        cv2_img = numpy.array(pil_img)
        cv2_img = cv2_img[:, :, ::-1].copy()  # Convert RGB to BGR

        rectangles = fa.get_detected_rectangles(pil_img)
        print(len(rectangles))

        for rect in rectangles:
            x1, y1, x2, y2, label_idx, score = rect
            # labelName = GTSRBDataset.get_label_names()[int(label_idx)]
            cv2_img = cv2.rectangle(cv2_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            # cv2_img = cv2.putText(cv2_img, labelName, (int(x1), int(y1) + 8), font, fontScale, color, 1, cv2.LINE_AA)
            cv2_img = cv2.putText(cv2_img, "{:.3f}".format(score), (int(x1), int(y1) + 16), font, fontScale, color, 1,
                                  cv2.LINE_AA)

        cv2.imshow('img', cv2_img)
        cv2.waitKey(0)


def test_fx():
    folder = os.path.dirname(__file__) + '/../test/'
    from PIL import Image

    bs = [10, 20, 30, 40, 50, 60, 70]
    cs = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    # x = 28
    # y = 28

    # cvs = Image.new('RGB', (x * len(bs), y * len(cs)))

    for b in bs:
        for c in cs:
            filename = '{}x{}_out_sign{}.jpg'

            im = Image.open(folder + filename.format(b, c, ""))

            im = (NormalizeHist())(im)

            im.save(folder + filename.format(b, c, "_O"))
            (Rgb2Hsl())(im).save(folder + filename.format(b, c, "_Z"))


def test_unfold(kw=2, kh=2):
    im = Image.open(os.path.dirname(__file__) + '/../test/test.jpg').convert("RGB")
    im.resize((100, 100), resample=Image.BOX).save(os.path.dirname(__file__) + '/../test/test_big.jpg')
    imt = (ToTensor())(im)
    # imt = imt.permute(2, 1, 0)
    c, w, h = imt.shape

    # works
    imt = imt.view(c, w, h).detach()
    patches = imt.permute(1, 2, 0) \
        .unfold(0, 2, 1) \
        .unfold(1, 2, 1) \
        .reshape(-1, c, kw, kh)

    positions = list(range(0, 9))
    for patch, pos in zip(patches, positions):
        # print(patch)
        # patch = patch.view(c, kw, kh)
        # patch = patch.permute(2, 1, 0)
        (ToPILImage())(patch).resize((100, 100), resample=Image.BOX).save(
            os.path.dirname(__file__) + '/../test/test_{}.jpg'.format(pos))


if __name__ == '__main__':

    # test_unfold()
    # exit(1)

    # test_fx()
    # exit(1)

    train_classifier()
    exit(1)

    # train_detector()
    # exit(1)

    # bootstrap()
    #train_detector_bootstrap()
    #exit(1)

    eval_stream(0)
    exit(1)

    # eval_stream(os.path.dirname(__file__) + '/video.mp4')
    # eval_img(os.path.dirname(__file__) + '/../street_sign_eval/wb.jpg')
    # exit(1)
    # load image, any size

    parser = argparse.ArgumentParser(description='Tries the models.')

    parser.add_argument('files', metavar='Files', type=str, nargs='+', help='image files')

    parser.add_argument('--sw', help='run the sliding window')

    args = parser.parse_args()
    print(args)
    print(args.files)
    if args.sw is None:
        # no sliding window
        pass
    else:
        # sliding window
        eval_stream()
