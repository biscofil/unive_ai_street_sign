class DatasetAugmentator:

    def __init__(self, dataset: list, operations: list):
        self.dataset = dataset
        self.operations = operations

    def __len__(self):
        return len(self.dataset) * len(self.operations)

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]
