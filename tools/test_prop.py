# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import __init_paths

import argparse
import os

import cv2
import torch
import numpy as np


from pysot.core.config import cfg
from pysot.models.model_2021 import Model2021
from pysot.tracker.tracker_builder_2021 import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', default='UAV123', type=str,
        help='datasets')
parser.add_argument('--config', default='../experiments/siamrpn_r50_l234_dwxcorr/config.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='../snapshot/three_ds_both/occl_iou.pth', type=str,
        help='snapshot of models to eval') #six_ds_l1_iou10/three_ds_occl/
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--results', default='../results', type=str,
        help='snapshot of models to eval')
# 12 book1 13 butterfly 15 conduction1 19 drone1 26 flamingo1 28 girl
# 41 leaves 47 rabbit 54 soccer1 60 zebrafish1
parser.add_argument('--vis',  action='store_true',
        help='whether visualzie result')
args = parser.parse_args()
torch.set_num_threads(1)
#['ants1', 'ants3', 'bag', 'ball1', 'ball2', 'basketball'
#, 'birds1', 'blanket', 'bmx', 'bolt1', 'bolt2', 'book',
# 'butterfly', 'car1', 'conduction1', 'crabs1', 'crossing',
# 'dinosaur', 'drone_across', 'drone_flip', 'drone1', 'fernando', 
#'fish1', 'fish2', 'fish3', 'flamingo1', 'frisbee', 'girl', 'glove', 
#'godfather', 'graduate', 'gymnastics1', 'gymnastics2', 'gymnastics3', 
#'hand', 'handball1', 'handball2', 'helicopter', 'iceskater1', 
#'iceskater2', 'leaves', 'matrix', 'motocross1', 'motocross2', 
#'nature', 'pedestrian1', 'rabbit', 'racing', 'road', 'shaking',
# 'sheep', 'singer2', 'singer3', 'soccer1', 'soccer2', 'soldier', 
#'tiger', 'traffic', 'wiper', 'zebrafish1']

overlaps2 = []
occl2 = [] # rabbit
#wiper, soldier, soccer1, rabbit, 
vars2 = []
mode = 0 ############################################################################
def main():
    # load config
    cfg.merge_from_file(args.config)
    
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)
    
    # create model
    model = Model2021()
    
    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()
    
    # build tracke
    tracker = build_tracker(model)
    
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    
    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0 
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            overlaps1 = []
            vars1 = []
            vars0 = []
            occl1 = []
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
    
            frame_width = 768#img.shape[1]
            frame_height = 576#img.shape[0]
            video_loc = os.path.join('../results', model_name, video.name)
    
            out = cv2.VideoWriter(video_loc+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height),True)
            if video.tags['occlusion']==[] or (np.array(video.tags['occlusion'])==1).sum()==0:
                print("\t\tdiscard occlusion")
                continue
                video.tags['occlusion'] = video.tags['all']
    
            for idx, (img, gt_bbox) in enumerate(video):
                   
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()

                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    box1 = gt_bbox_
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                    if idx == 0:
                        print(img.shape)
                elif idx > frame_counter:
                    outputs = tracker.track(img, mode)
                    pred_bbox = outputs['bbox']
                    
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
    #######################################################################################
                    cx, cy, w, h  = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    box2 = gt_bbox_                
                    w1, h1 = box1[2], box1[3]
                    w2, h2 = box2[2], box2[3]
                    cx1, cy1 = (img.shape[1]//2, img.shape[0]//2)
                    cx2, cy2 = (box2[2]/2+box2[0], box2[3]/2+box2[1])
    #                box1 = box2
                    # scale variation
                    s1 = np.sqrt(w1*h1)
                    s2 = np.sqrt(w2*h2)    
                    sv = max(s1/s2, s2/s1)
                    
                    # aspect ratio variation
                    r1, r2 = h1/w1, h2/w2
                    arv = max(r1/r2, r2/r1)
                    
                    # fast motion
                    fm = np.sqrt((cx2-cx1)**2+(cy2-cy1)**2)/np.sqrt(s1*s2)
                    vars0.append(np.array([sv, arv, fm, outputs['cls2']]))
                    # occlusion
    #########################################################################################
     #               print(idx, outputs['var'], np.array([sv, arv, fm]))  ##################################
                    overlaps1.append(overlap)
                    vars1.append(outputs['cls2'])
                    if idx<=len(video.tags['occlusion']):
                        occl1.append(video.tags['occlusion'][idx])
                    else:
                        occl1.append(np.zeros(idx-len(video.tags['occlusion'])))
                    if overlap > 0.0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        print("-------loss---------")
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                        for l in range(0,5):
                            vars1.append(-0.2)
                            occl1.append(video.tags['occlusion'][idx+l])
                else:
                    pred_bboxes.append(0)
                    
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (255, 0, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    print(idx)
#                    cv2.putText(img, 'occl_gt:'+str(video.tags['occlusion'][idx-1]), (240, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(img, 'proposed_TL:'+str(lost_number), (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
#                    cv2.putText(img, 'occl_pred:'+str(vars1[idx-1]), (240, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    out.write(img)
                    cv2.imwrite(video_loc+'_prop_'+str(idx)+'.png',img)
                    cv2.waitKey(1)
    
            toc /= cv2.getTickFrequency()
            # save results
            out.release()
            video_path = os.path.join(args.results, args.dataset, model_name,
                    'baseline', video.name)
    
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}, mIOU: {:0.4f}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number, np.array(overlaps1).mean()))
    #        plt.plot(overlaps1)
    #        plt.plot(np.array(vars0)[:,3])
  #          plt.plot(np.array(occl1))

  #          plt.plot(np.array(vars1))
    #        print(np.correlate(overlaps1,np.array(vars1)[:,2]))
            overlaps2.append(np.array(overlaps1).mean())
            occl2.append(np.array(occl1))
            vars2.append(np.array(vars1))
 #           if args.video != '':
#                v_idx=0
#            print(100*(confusion_matrix(occl2[v_idx],vars2[v_idx]).ravel()))
                
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    
    # cv2.destroyAllWindows()
    # print("Total Mean IOU is   %0.4f"%np.array(overlaps2).mean())

    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('../results', args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('../results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join('../results', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc ))

main()
cv2.destroyAllWindows()
#print("Total Mean IOU is   %0.4f"%np.array(overlaps2).mean())
# if __name__ == '__main__':
#     main()
#     cv2.destroyAllWindows()
