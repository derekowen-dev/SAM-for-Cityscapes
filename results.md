IoU : intersection over union
mIoU : mean intersection over union


RUN 0: box-prompt zero-shot SAM baseline
    - per-class IoU
        - road:         0.8544
        - sidewalk:     0.6984
        - building:     0.6944
        - person:       0.6114
        - car:          0.7374
    -                               mIoU: 0.7192


RUN 1: SAM-based Cityscapes-tuned semantic head
    - details:
        - batch size:           1
        - epochs:               3
        - max train samples:    800
        - max val samples:      200

    - per-class IoU
        - road:         0.9399
        - sidewalk:     0.6206
        - building:     0.8008
        - person:       0.6017
        - car:          0.7843
    -                               mIoU: 0.7495

RUN 2: SAM-based Cityscapes-tuned semantic head
    - details:
        - batch size:           2
        - epochs:               5
        - max train samples:    2975 (all)
        - max val samples:      500 (all)
        
    - per-class IoU
        - road:         0.9256
        - sidewalk:     0.6990
        - building:     0.8535
        - person:       0.6555
        - car:          0.8731
    -                               mIoU: 0.8013