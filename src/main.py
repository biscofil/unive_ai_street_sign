import cv2
import torch

from src.BootStrapper import BootStrapper
from src.FrameAnalyzer import FrameAnalyzer
from src.networks.CnnClassifier import CnnClassifier
from src.networks.CnnDetector import CnnDetector
from src.networks.CnnDetectorWithSlidingWindow import CnnDetectorWithSlidingWindow

device = "cuda:0" if torch.cuda.is_available() else "cpu"

seed = 852144

if 'cuda' in device:
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)


def train_classifier():
    cnn_classifier_model = CnnClassifier(device)
    # cnn_classifier_model.load_from_file()
    cnn_classifier_model.train_model()
    cnn_classifier_model.store_to_file()


def train_detector():
    cnn_detector_model = CnnDetector(device)
    cnn_detector_model.load_from_file()
    cnn_detector_model.train_model()
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


def train_from_stream() -> None:
    """
    The feed must have anything but street signs
    Save wrongly classified patches
    :return:
    """
    cnn_detector_model = CnnDetector('cpu')
    cnn_detector_model.load_from_file()
    fa = FrameAnalyzer(cnn_detector_model, 'cpu', train=True)  # device
    # fa.video_analyzer(os.path.dirname(__file__) + '/video.mp4')
    fa.video_analyzer(0)  # webcam
    train_detector()


def eval_stream(stream=0) -> None:
    # strean = os.path.dirname(__file__) + '/video.mp4'
    cnn_sw_detector_model = CnnDetectorWithSlidingWindow(device)
    cnn_sw_detector_model.load_from_file()

    cap = cv2.VideoCapture(stream)
    # fetch first frame
    _, img = cap.read()

    fa = FrameAnalyzer(cnn_sw_detector_model, device, train=False, width=img.shape[1], height=img.shape[0])  # device

    while True:
        _, img = cap.read()

        if img is not None:
            rectangles = fa.get_detected_rectangles(img)
            print(len(rectangles))

            for rect in rectangles:
                x1, y1, x2, y2, score = rect
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)

            cv2.imshow('img', img)
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # train_detector()
    # train_classifier()
    # bootstrap()
    # train_from_stream()
    eval_stream()
    # load image, any size
