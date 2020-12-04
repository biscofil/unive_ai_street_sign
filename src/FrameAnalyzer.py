import cv2
from torch import nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor

from src.custom_transformations.CustomRescale import CustomRescale
from src.networks.FullNetwork import FullNetwork


class FrameAnalyzer:

    def __init__(self, model: FullNetwork, device: str, scale: float = 1.3, step_size: int = 15):
        self.model = model
        self.device = device
        self.scale = scale
        self.step_size = step_size
        self.tensor_transformation = ToTensor()
        self.sm = nn.Softmax(dim=1)

        self.win_w = model.img_size
        self.win_h = self.win_w

    @staticmethod
    def pyramid(image, scale=1.5, minSize=(35, 35)):
        # yield the original image
        yield image
        # keep looping over the pyramid
        while True:
            # compute the new dimensions of the image and resize it
            w = int(image.shape[1] / scale)
            image = cv2.resize(image, (w, w))
            # if the resized image does not meet the supplied minimum
            # size, then stop constructing the pyramid
            if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
                break
            # yield the next image in the pyramid
            yield image

    @staticmethod
    def sliding_window(image, stepSize, windowSize):
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield x, y, image[y:y + windowSize[1], x:x + windowSize[0]]

    def video_analyzer(self):

        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture('car_dash.mp4')

        while True:
            _, img = cap.read()

            rectangles = self.frame_analyzer(img)

            cv2.imshow('img', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cap.release()

    def frame_analyzer(self, img) -> list:
        gray = img  # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        out = []

        # loop over the image pyramid
        for resized in FrameAnalyzer.pyramid(gray, scale=self.scale):
            # loop over the sliding window for each layer of the pyramid
            for (x, y, window) in FrameAnalyzer.sliding_window(resized, stepSize=self.step_size,
                                                               windowSize=(self.win_w, self.win_h)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != self.win_h or window.shape[1] != self.win_w:
                    continue

                dist = self.get_patch_label(window)

                if float(dist.max()) > 0.95:  # TODO fix
                    # print(float(output[0][index]), float(dist[0][index]))

                    # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
                    # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
                    # WINDOW
                    # since we do not have a classifier, we'll just draw the window
                    # clone = resized.copy()
                    # cv2.rectangle(gray, (x, y), (x + self.win_w, y + self.win_h), (0, 255, 0), 2)
                    # cv2.imshow("Window", clone)
                    # cv2.waitKey(1)
                    # time.sleep(0.1)
                    out.append((x, y, x + self.win_w, y + self.win_h))
        return out

    def get_patch_label(self, window) -> int:
        # scale
        cs = CustomRescale(self.model.img_size)
        window = cs({
            "image" : window
        })
        # get tensor
        self.tensor_transformation(window)
        image_tensor = self.tensor_transformation(window).float()
        image_tensor = image_tensor.unsqueeze_(0)  # .reshape(-1)
        # run model
        input = Variable(image_tensor)
        input = input.to(self.device)
        output = self.model(input)
        dist = self.sm(output)
        return dist.data.cpu().numpy().argmax()
