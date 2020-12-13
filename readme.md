# Image and Video Understanding + Artificial Intelligence



##TODO

*[ ] Check test / evaluation partitioning, remove eval parameter
*[ ] Right now we apply augmentation before cropping and scaling. We must allow to apply augmentation at any point of the transformation pipeline 

## Bootstrapping for negative examples

Start with a small set of non face examples of the training set
train the neural network
run the detector on random images, collecting all random images classified as faces (false positives) and add those to the training set. Repeat.
 
## Setup

Download the dataset from https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

# TODO update
Place the dataset files in the `src/dataset/GTSRB-Training_fixed` in order 
to have file paths like `src/dataset/GTSRB-Training_fixed/GTSRB/Training/00000/00000_00000.ppm`