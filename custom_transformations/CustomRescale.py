from torchvision import transforms


class CustomRescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def getNewSize(self, sample):
        new_h, new_w = self.output_size, self.output_size
        if False:
            h, w = sample["image"].shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size
            new_h, new_w = int(new_h), int(new_w)
        return new_h, new_w

    def __call__(self, sample: dict):

        new_h, new_w = self.getNewSize(sample)

        img = sample["image"].resize((new_h, new_w))

        return img
