BASE_PATH = "/home/scarrion/datasets/covid19/front"
OUTPUT_PATH = "/home/scarrion/projects/mltests/covid19/code/.outputs"
BACKBONE = 'resnet34'
EPOCHS1 = 1
EPOCHS2 = 1
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
SEED = 1234
SPLITS = (0.8, 0.1, 0.1)
assert sum(SPLITS) == 1.0
