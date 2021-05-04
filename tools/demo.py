from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import __init_paths
import os
import argparse

import torch
import numpy as np
from glob import glob
import cv2
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.model_load import load_pretrain

import gc
torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config',default='../experiments/siamrpn_r50_l234_dwxcorr/config.yaml',type=str, help='config file')
parser.add_argument('--snapshot',default='../snapshot/rpnpp_iou_vot/checkpoint_e39.pth', type=str, help='model name')
parser.add_argument('--video_name', default='helicopter.avi', type=str,
                    help='videos or image files')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
#    model.load_state_dict(torch.load(args.snapshot,
#        map_location=lambda storage, loc: storage.cpu()))
    model = load_pretrain(model, args.snapshot).cuda().eval()
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    frame_width = int(848)
    frame_height = int(480)
    out = cv2.VideoWriter(video_name+'_out'+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height),True)
    for frame in get_frames(args.video_name):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 0, 255), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 0, 255), 3)
            cv2.imshow(video_name, frame)
            out.write(frame)
            cv2.waitKey(40)
    cv2.destroyAllWindows()
    gc.collect()    

if __name__ == '__main__':
    main()
