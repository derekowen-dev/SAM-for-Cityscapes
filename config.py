import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = PROJECT_ROOT

SAM_CHECKPOINT = os.path.join(PROJECT_ROOT, "sam_vit_h_4b8939.pth")
HEAD_WEIGHTS = os.path.join(PROJECT_ROOT, "out_models_cityscapes_head", "sam_cityscapes_head.pth")

OUT_EXAMPLES_MAIN = os.path.join(PROJECT_ROOT, "out_examples_main")
OUT_MODELS_DIR = os.path.dirname(HEAD_WEIGHTS)
