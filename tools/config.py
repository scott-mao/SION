META_ARC: "siamrpn_r50_l234_dwxcorr_otb_8gpu"

BACKBONE:
    TYPE: "resnet50"
    KWARGS:
        used_layers: [2, 3, 4]
    PRETRAINED: 'pretrained_models/resnet50.model'
    TRAIN_LAYERS: ['layer1', 'layer2', 'layer3', 'layer4']
    TRAIN_EPOCH: 5
    LAYERS_LR: 0.1

ADJUST:
    ADJUST: true
    TYPE: "AdjustAllLayer"
    KWARGS:
        in_channels: [512, 1024, 2048]
        out_channels: [256, 256, 256]

RPN:
    TYPE: 'MultiRPN'
    KWARGS:
        anchor_num: 5
        in_channels: [256, 256, 256]
        weighted: false 

MASK:
    MASK: false

ANCHOR:
    STRIDE: 8
    RATIOS: [0.33, 0.5, 1, 2, 3]
    SCALES: [8]
    ANCHOR_NUM: 5

TRACK:
    TYPE: 'SiamRPNTracker'
    PENALTY_K: 0.24
    WINDOW_INFLUENCE: 0.5
    LR: 0.25
    EXEMPLAR_SIZE: 127
    INSTANCE_SIZE: 255
    BASE_SIZE: 8
    CONTEXT_AMOUNT: 0.5

TRAIN:
    EPOCH: 20
    START_EPOCH: 0
    BATCH_SIZE: 16
    BASE_LR: 0.005
    CLS_WEIGHT: 1.0
    LOC_WEIGHT: 1.2
    RESUME: ''

    LR:
        TYPE: 'log'
        KWARGS:
            start_lr: 0.005
            end_lr: 0.0025
    LR_WARMUP:
        TYPE: 'step'
        EPOCH: 5
        KWARGS:
            start_lr: 0.001
            end_lr: 0.005
            step: 1

DATASET:
    NAMES: 
    - 'VID'
    #- 'YOUTUBEBB'
    - 'COCO'
    - 'DET'

    TEMPLATE:
        SHIFT: 4
        SCALE: 0.05
        BLUR: 0.0
        FLIP: 0.0
        COLOR: 1.0

    SEARCH:
        SHIFT: 64
        SCALE: 0.18
        BLUR: 0.2
        FLIP: 0.0
        COLOR: 1.0

    NEG: 0.2
    GRAY: 0.2
