# Segment Anything on Cityscapes (CSCI 490 DPL Final Project)

This repo evaluates Segment Anything (SAM) on Cityscapes urban driving scenes and trains a Cityscapes-tuned semantic head on top of SAM

# setup

download Cityscapes dataset (gtFine, leftImg8bit) -- https://www.cityscapes-dataset.com/
    - keep in the same directory as scripts

download SAM ViT-H checkpoint (sam_vit_h_4b8939.pth) -- https://github.com/facebookresearch/segment-anything.git
    - keep in the same directory as scripts


# commands to get started

conda create -n sam_city python=3.10
conda activate sam_city

pip install -r requirements.txt

git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
cd ..

python zero_shot_eval_box_prompts.py    -- box-prompt zero-shot SAM evaluation

python train_sam_cityscapes_head.py     -- trains and evaluates SAM-based Cityscapes-tuned semantic head (generates sam_cityscapes_head.pth in out_models_cityscapes_head)


python save_examples_main.py            -- generates 5 random collages of all steps (6 images) (requires sam_cityscapes_head.pth in out_models_cityscapes_head)
