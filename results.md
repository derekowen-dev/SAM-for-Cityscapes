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

Output from RUN 2:
Epoch 1 Step 50/1488 avg_loss_per_pixel 0.588508
Epoch 1 Step 100/1488 avg_loss_per_pixel 0.499727 
Epoch 1 Step 150/1488 avg_loss_per_pixel 0.446841 
Epoch 1 Step 200/1488 avg_loss_per_pixel 0.412855 
Epoch 1 Step 250/1488 avg_loss_per_pixel 0.390926 
Epoch 1 Step 300/1488 avg_loss_per_pixel 0.376469 
Epoch 1 Step 350/1488 avg_loss_per_pixel 0.367946 
Epoch 1 Step 400/1488 avg_loss_per_pixel 0.360434 
Epoch 1 Step 450/1488 avg_loss_per_pixel 0.353772 
Epoch 1 Step 500/1488 avg_loss_per_pixel 0.349086 
Epoch 1 Step 550/1488 avg_loss_per_pixel 0.346694 
Epoch 1 Step 600/1488 avg_loss_per_pixel 0.343795 
Epoch 1 Step 650/1488 avg_loss_per_pixel 0.339844 
Epoch 1 Step 700/1488 avg_loss_per_pixel 0.334753 
Epoch 1 Step 750/1488 avg_loss_per_pixel 0.328949 
Epoch 1 Step 800/1488 avg_loss_per_pixel 0.324034 
Epoch 1 Step 850/1488 avg_loss_per_pixel 0.322123 
Epoch 1 Step 900/1488 avg_loss_per_pixel 0.318630 
Epoch 1 Step 950/1488 avg_loss_per_pixel 0.316264 
Epoch 1 Step 1000/1488 avg_loss_per_pixel 0.313449 
Epoch 1 Step 1050/1488 avg_loss_per_pixel 0.310489 
Epoch 1 Step 1100/1488 avg_loss_per_pixel 0.308194 
Epoch 1 Step 1150/1488 avg_loss_per_pixel 0.304907 
Epoch 1 Step 1200/1488 avg_loss_per_pixel 0.302948 
Epoch 1 Step 1250/1488 avg_loss_per_pixel 0.299829 
Epoch 1 Step 1300/1488 avg_loss_per_pixel 0.297200 
Epoch 1 Step 1350/1488 avg_loss_per_pixel 0.296146 
Epoch 1 Step 1400/1488 avg_loss_per_pixel 0.293350 
Epoch 1 Step 1450/1488 avg_loss_per_pixel 0.292280 
Epoch 1 Step 1488/1488 avg_loss_per_pixel 0.291400 
Epoch 1/5 Train loss per labeled pixel: 0.291400 
Val per-class IoU: 
class 7: 0.9031 
class 8: 0.6201 
class 11: 0.8319 
class 24: 0.5955 
class 26: 0.8376 
Val mIoU: 0.7576 
Epoch 2 Step 50/1488 avg_loss_per_pixel 0.286837 
Epoch 2 Step 100/1488 avg_loss_per_pixel 0.273782 
Epoch 2 Step 150/1488 avg_loss_per_pixel 0.269978 
Epoch 2 Step 200/1488 avg_loss_per_pixel 0.271480 
Epoch 2 Step 250/1488 avg_loss_per_pixel 0.271570 
Epoch 2 Step 300/1488 avg_loss_per_pixel 0.265506 
Epoch 2 Step 350/1488 avg_loss_per_pixel 0.261109 
Epoch 2 Step 400/1488 avg_loss_per_pixel 0.250925 
Epoch 2 Step 450/1488 avg_loss_per_pixel 0.246296 
Epoch 2 Step 500/1488 avg_loss_per_pixel 0.248133 
Epoch 2 Step 550/1488 avg_loss_per_pixel 0.246697 
Epoch 2 Step 600/1488 avg_loss_per_pixel 0.247766 
Epoch 2 Step 650/1488 avg_loss_per_pixel 0.248938 
Epoch 2 Step 700/1488 avg_loss_per_pixel 0.246973 
Epoch 2 Step 750/1488 avg_loss_per_pixel 0.246832 
Epoch 2 Step 800/1488 avg_loss_per_pixel 0.245299 
Epoch 2 Step 850/1488 avg_loss_per_pixel 0.244103 
Epoch 2 Step 900/1488 avg_loss_per_pixel 0.244939 
Epoch 2 Step 950/1488 avg_loss_per_pixel 0.244769 
Epoch 2 Step 1000/1488 avg_loss_per_pixel 0.244991 
Epoch 2 Step 1050/1488 avg_loss_per_pixel 0.244961 
Epoch 2 Step 1100/1488 avg_loss_per_pixel 0.245023 
Epoch 2 Step 1150/1488 avg_loss_per_pixel 0.244711 
Epoch 2 Step 1200/1488 avg_loss_per_pixel 0.243494 
Epoch 2 Step 1250/1488 avg_loss_per_pixel 0.241896 
Epoch 2 Step 1300/1488 avg_loss_per_pixel 0.240935 
Epoch 2 Step 1350/1488 avg_loss_per_pixel 0.241324 
Epoch 2 Step 1400/1488 avg_loss_per_pixel 0.241160 
Epoch 2 Step 1450/1488 avg_loss_per_pixel 0.240329 
Epoch 2 Step 1488/1488 avg_loss_per_pixel 0.239644 
Epoch 2/5 Train loss per labeled pixel: 0.239644 
Val per-class IoU: 
class 7: 0.9182 
class 8: 0.6869 
class 11: 0.8382 
class 24: 0.6162 
class 26: 0.8402 
Val mIoU: 0.7800 
Epoch 3 Step 50/1488 avg_loss_per_pixel 0.199681 
Epoch 3 Step 100/1488 avg_loss_per_pixel 0.197069 
Epoch 3 Step 150/1488 avg_loss_per_pixel 0.198930 
Epoch 3 Step 200/1488 avg_loss_per_pixel 0.202873 
Epoch 3 Step 250/1488 avg_loss_per_pixel 0.209199 
Epoch 3 Step 300/1488 avg_loss_per_pixel 0.206460 
Epoch 3 Step 350/1488 avg_loss_per_pixel 0.217756 
Epoch 3 Step 400/1488 avg_loss_per_pixel 0.221624 
Epoch 3 Step 450/1488 avg_loss_per_pixel 0.220059 
Epoch 3 Step 500/1488 avg_loss_per_pixel 0.216053 
Epoch 3 Step 550/1488 avg_loss_per_pixel 0.214827 
Epoch 3 Step 600/1488 avg_loss_per_pixel 0.213552 
Epoch 3 Step 650/1488 avg_loss_per_pixel 0.212863 
Epoch 3 Step 700/1488 avg_loss_per_pixel 0.213556 
Epoch 3 Step 750/1488 avg_loss_per_pixel 0.218776 
Epoch 3 Step 800/1488 avg_loss_per_pixel 0.218700 
Epoch 3 Step 850/1488 avg_loss_per_pixel 0.220914 
Epoch 3 Step 900/1488 avg_loss_per_pixel 0.221012 
Epoch 3 Step 950/1488 avg_loss_per_pixel 0.221404 
Epoch 3 Step 1000/1488 avg_loss_per_pixel 0.220769 
Epoch 3 Step 1050/1488 avg_loss_per_pixel 0.222176 
Epoch 3 Step 1100/1488 avg_loss_per_pixel 0.224130 
Epoch 3 Step 1150/1488 avg_loss_per_pixel 0.224457 
Epoch 3 Step 1200/1488 avg_loss_per_pixel 0.224279 
Epoch 3 Step 1250/1488 avg_loss_per_pixel 0.225467 
Epoch 3 Step 1300/1488 avg_loss_per_pixel 0.225480 
Epoch 3 Step 1350/1488 avg_loss_per_pixel 0.224739 
Epoch 3 Step 1400/1488 avg_loss_per_pixel 0.223700 
Epoch 3 Step 1450/1488 avg_loss_per_pixel 0.223419 
Epoch 3 Step 1488/1488 avg_loss_per_pixel 0.223307 
Epoch 3/5 Train loss per labeled pixel: 0.223307 
Val per-class IoU: 
class 7: 0.9224 
class 8: 0.6711 
class 11: 0.8467 
class 24: 0.6370 
class 26: 0.8529 
Val mIoU: 0.7860 
Epoch 4 Step 50/1488 avg_loss_per_pixel 0.202789 
Epoch 4 Step 100/1488 avg_loss_per_pixel 0.205861 
Epoch 4 Step 150/1488 avg_loss_per_pixel 0.221343 
Epoch 4 Step 200/1488 avg_loss_per_pixel 0.225189 
Epoch 4 Step 250/1488 avg_loss_per_pixel 0.229405 
Epoch 4 Step 300/1488 avg_loss_per_pixel 0.222999 
Epoch 4 Step 350/1488 avg_loss_per_pixel 0.220797 
Epoch 4 Step 400/1488 avg_loss_per_pixel 0.219039 
Epoch 4 Step 450/1488 avg_loss_per_pixel 0.216009 
Epoch 4 Step 500/1488 avg_loss_per_pixel 0.217449 
Epoch 4 Step 550/1488 avg_loss_per_pixel 0.215229 
Epoch 4 Step 600/1488 avg_loss_per_pixel 0.215009 
Epoch 4 Step 650/1488 avg_loss_per_pixel 0.215416 
Epoch 4 Step 700/1488 avg_loss_per_pixel 0.214588 
Epoch 4 Step 750/1488 avg_loss_per_pixel 0.214061 
Epoch 4 Step 800/1488 avg_loss_per_pixel 0.212719 
Epoch 4 Step 850/1488 avg_loss_per_pixel 0.212156 
Epoch 4 Step 900/1488 avg_loss_per_pixel 0.214492 
Epoch 4 Step 950/1488 avg_loss_per_pixel 0.214163 
Epoch 4 Step 1000/1488 avg_loss_per_pixel 0.213462 
Epoch 4 Step 1050/1488 avg_loss_per_pixel 0.213429 
Epoch 4 Step 1100/1488 avg_loss_per_pixel 0.213323 
Epoch 4 Step 1150/1488 avg_loss_per_pixel 0.212939 
Epoch 4 Step 1200/1488 avg_loss_per_pixel 0.213773 
Epoch 4 Step 1250/1488 avg_loss_per_pixel 0.213459 
Epoch 4 Step 1300/1488 avg_loss_per_pixel 0.213851 
Epoch 4 Step 1350/1488 avg_loss_per_pixel 0.214914 
Epoch 4 Step 1400/1488 avg_loss_per_pixel 0.215063 
Epoch 4 Step 1450/1488 avg_loss_per_pixel 0.215818 
Epoch 4 Step 1488/1488 avg_loss_per_pixel 0.215986 
Epoch 4/5 Train loss per labeled pixel: 0.215986 
Val per-class IoU: 
class 7: 0.9244 
class 8: 0.7087 
class 11: 0.8392 
class 24: 0.6476 
class 26: 0.8654 
Val mIoU: 0.7970 
Epoch 5 Step 50/1488 avg_loss_per_pixel 0.219674 
Epoch 5 Step 100/1488 avg_loss_per_pixel 0.226089 
Epoch 5 Step 150/1488 avg_loss_per_pixel 0.213289 
Epoch 5 Step 200/1488 avg_loss_per_pixel 0.214409 
Epoch 5 Step 250/1488 avg_loss_per_pixel 0.216760 
Epoch 5 Step 300/1488 avg_loss_per_pixel 0.216434 
Epoch 5 Step 350/1488 avg_loss_per_pixel 0.215680 
Epoch 5 Step 400/1488 avg_loss_per_pixel 0.215134 
Epoch 5 Step 450/1488 avg_loss_per_pixel 0.213682 
Epoch 5 Step 500/1488 avg_loss_per_pixel 0.213475 
Epoch 5 Step 550/1488 avg_loss_per_pixel 0.211951 
Epoch 5 Step 600/1488 avg_loss_per_pixel 0.215239 
Epoch 5 Step 650/1488 avg_loss_per_pixel 0.214910 
Epoch 5 Step 700/1488 avg_loss_per_pixel 0.215229 
Epoch 5 Step 750/1488 avg_loss_per_pixel 0.214017 
Epoch 5 Step 800/1488 avg_loss_per_pixel 0.215720 
Epoch 5 Step 850/1488 avg_loss_per_pixel 0.213524 
Epoch 5 Step 900/1488 avg_loss_per_pixel 0.214150 
Epoch 5 Step 950/1488 avg_loss_per_pixel 0.214369 
Epoch 5 Step 1000/1488 avg_loss_per_pixel 0.214437 
Epoch 5 Step 1050/1488 avg_loss_per_pixel 0.213752 
Epoch 5 Step 1100/1488 avg_loss_per_pixel 0.212260 
Epoch 5 Step 1150/1488 avg_loss_per_pixel 0.211780 
Epoch 5 Step 1200/1488 avg_loss_per_pixel 0.211087 
Epoch 5 Step 1250/1488 avg_loss_per_pixel 0.211553 
Epoch 5 Step 1300/1488 avg_loss_per_pixel 0.211236 
Epoch 5 Step 1350/1488 avg_loss_per_pixel 0.210345 
Epoch 5 Step 1400/1488 avg_loss_per_pixel 0.210531 
Epoch 5 Step 1450/1488 avg_loss_per_pixel 0.209735 
Epoch 5 Step 1488/1488 avg_loss_per_pixel 0.209027 
Epoch 5/5 Train loss per labeled pixel: 0.209027 
Val per-class IoU: 
class 7: 0.9256 
class 8: 0.6990 
class 11: 0.8535 
class 24: 0.6555 
class 26: 0.8731 
Val mIoU: 0.8013