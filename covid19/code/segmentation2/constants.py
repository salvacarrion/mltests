BASE_PATH = "/home/scarrion/datasets/covid19/front"
OUTPUTS_PATH = ".outputs"
MODELS_PATH = "/home/scarrion/projects/mltests/covid19/code/.outputs/models"
MODELS_NAME = "model_last.h5"
BACKBONE = 'resnet34'
EPOCHS1 = 3
EPOCHS2 = 10
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
SEED = 1234
SPLITS = (0.8, 0.1, 0.1)
assert sum(SPLITS) == 1.0
