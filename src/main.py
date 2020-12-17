import os
import random

import cv2
import numpy
import torch
from PIL import Image

from src.FrameAnalyzer import FrameAnalyzer
from src.dataset.GTSRBDataset import GTSRBDataset
from src.networks.CnnClassifier import CnnClassifier

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
    cnn_classifier_model.train_model(max_epochs=100)
    cnn_classifier_model.store_to_file()


def train_classifier_bootstrap(iterations: int = 3, files_per_iteration: int = 50,
                               test_partition: float = 0.2) -> None:
    """
    The feed must have anything but street signs
    Save wrongly classified patches
    :return:
    """

    classifier = CnnClassifier(device).to(device)
    classifier.load_from_file()
    classifier = classifier.to(device)

    full_res_folder = os.path.dirname(__file__) + '/dataset/GTSRB_Negative/bootstrap_full_res/'
    output_folder = os.path.dirname(__file__) + '/dataset/GTSRB_Negative/images/'
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
            fa = FrameAnalyzer(device, classifier=classifier, width=pil_img.size[0], height=pil_img.size[1])
            if pil_img is not None:
                rectangles = fa.get_detected_street_signs(pil_img, limit=50)
                print("\t\t{} patches classified as street signs".format(len(rectangles)))
                if len(rectangles):
                    dirty = True
                    for rect in rectangles:
                        x1, y1, x2, y2, label_idx, score = rect
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        fn = '{}_{}_'.format(x1, y1) + jpg_file_name
                        # crop and save
                        pil_img.crop((x1, y1, x2, y2)).save(output_folder + fn)
                        new_files.append(fn)

        if dirty:
            print("Adding", len(new_files), "negative examples")
            # shuffle and partition into training and test
            random.shuffle(new_files)
            n_test = int(test_partition * len(new_files))
            new_files_test = new_files[:n_test]
            new_files_training = new_files[n_test:]
            GTSRBDataset.add_images_to_negative_examples_json(new_files_test, overwrite=False, test=True)
            GTSRBDataset.add_images_to_negative_examples_json(new_files_training, overwrite=False, test=False)
            # train the model
            # epochs = 2  # max(1, int(iterations / 3))
            classifier.train_model(max_epochs=20, max_patience=5)
            classifier.store_to_file()


def eval_stream(stream, font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 255, 255)) -> None:
    classifier = CnnClassifier(device)
    classifier.load_from_file()

    cap = cv2.VideoCapture(stream)
    _, cv2_img = cap.read()  # fetch first frame
    fa = FrameAnalyzer(device, width=cv2_img.shape[1], height=cv2_img.shape[0])

    while True:
        _, cv2_img = cap.read()

        if cv2_img is not None:

            pil_img = Image.fromarray(cv2_img)

            rectangles = fa.get_detected_street_signs(pil_img)
            print(len(rectangles))

            for rect in rectangles:
                x1, y1, x2, y2, label_idx, score = rect
                label_name = GTSRBDataset.get_label_names()[int(label_idx)]
                cv2_img = cv2.rectangle(cv2_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                cv2_img = cv2.putText(cv2_img, label_name, (int(x1), int(y1) + 8), font, fontScale, color, 1,
                                      cv2.LINE_AA)
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
    fa = FrameAnalyzer(device, width=pil_img.size[0], height=pil_img.size[1])  # device

    if pil_img is not None:

        cv2_img = numpy.array(pil_img)
        cv2_img = cv2_img[:, :, ::-1].copy()  # Convert RGB to BGR

        rectangles = fa.get_detected_street_signs(pil_img)
        print(len(rectangles))

        for rect in rectangles:
            x1, y1, x2, y2, label_idx, score = rect
            label_name = GTSRBDataset.get_label_names()[int(label_idx)]
            cv2_img = cv2.rectangle(cv2_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            cv2_img = cv2.putText(cv2_img, label_name, (int(x1), int(y1) + 8), font, fontScale, color, 1, cv2.LINE_AA)
            cv2_img = cv2.putText(cv2_img, "{:.3f}".format(score), (int(x1), int(y1) + 16), font, fontScale, color, 1,
                                  cv2.LINE_AA)

        cv2.imshow('img', cv2_img)
        cv2.waitKey(0)


if __name__ == '__main__':
    # train_classifier()

    # bootstrap()
    # train_classifier_bootstrap()

    # eval_stream(0)

    # eval_stream(os.path.dirname(__file__) + '/video.mp4')
    eval_img(os.path.dirname(__file__) + '/../street_sign_eval/wb.jpg')
