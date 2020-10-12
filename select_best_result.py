import torch
from torchvision import transforms
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.utils import *
import json
import mmcv

from shutil import copyfile
import numpy as np
import os
import ntpath
from tool.darknet2pytorch import *
from infer import infer
from tqdm import tqdm
from skimage import measure

import sys
sys.path.append('./mmdetection/')
from mmdet import __version__
from mmdet.apis import init_detector, inference_detector
from mmdet.models import iou_loss

def get_ssd_bbscore(model, filepath, gt_bboxes=None):
    show_score_thr = 0.3
    file_name = ntpath.basename(filepath)
    img_dir = 'ori_images/'

    result_p, _ = inference_detector(model, filepath, gt_bboxes=gt_bboxes)
    result_c, _ = inference_detector(model, os.path.join(img_dir, file_name), gt_bboxes=gt_bboxes)
    if isinstance(result_p, tuple):
        bbox_results, _ = result_p
        result_p = bbox_results
        bbox_results, _ = result_c
        result_c = bbox_results
    result_above_confidence_num_p = 0
    result_above_confidence_num_c = 0
    result_p = np.concatenate(result_p)
    result_c = np.concatenate(result_c)
    
    for ir in range(len(result_p)):
        if result_p[ir, 4] > show_score_thr:
            result_above_confidence_num_p = result_above_confidence_num_p + 1
    for ir in range(len(result_c)):
        if result_c[ir, 4] > show_score_thr:
            #x1, y1, x2, y2 = result_c[ir, :4]
            #with open('groundtruth/%s.txt'%fname, 'a') as f:
            #    f.write('%.2f %.2f %.2f %.2f\n' % (x1, y1, x2, y2))
            result_above_confidence_num_c = result_above_confidence_num_c + 1
    if(result_above_confidence_num_c == 0):
        return 0
    #print("Original: %d Attack: %d" % (result_above_confidence_num_c, result_above_confidence_num_p))
    bb_score = 1 - min(result_above_confidence_num_c,
                       result_above_confidence_num_p) / result_above_confidence_num_c

    return bb_score


def get_fasterrcnn_bbscore(model, filepath, gt_bboxes=None):
    show_score_thr = 0.3
    file_name = ntpath.basename(filepath)
    img_dir = 'ori_images/'

    result_p, _, rpn_loss = inference_detector(model, filepath, gt_bboxes=gt_bboxes)
    result_c, _, rpn_loss = inference_detector(model, os.path.join(img_dir, file_name), gt_bboxes=gt_bboxes)
    if isinstance(result_p, tuple):
        bbox_results, _ = result_p
        result_p = bbox_results
        bbox_results, _ = result_c
        result_c = bbox_results
    result_above_confidence_num_p = 0
    result_above_confidence_num_c = 0
    result_p = np.concatenate(result_p)
    result_c = np.concatenate(result_c)
    
    for ir in range(len(result_p)):
        if result_p[ir, 4] > show_score_thr:
            result_above_confidence_num_p = result_above_confidence_num_p + 1
    for ir in range(len(result_c)):
        if result_c[ir, 4] > show_score_thr:
            #x1, y1, x2, y2 = result_c[ir, :4]
            #with open('groundtruth/%s.txt'%fname, 'a') as f:
            #    f.write('%.2f %.2f %.2f %.2f\n' % (x1, y1, x2, y2))
            result_above_confidence_num_c = result_above_confidence_num_c + 1
    if(result_above_confidence_num_c == 0):
        return 0
    #print("Original: %d Attack: %d" % (result_above_confidence_num_c, result_above_confidence_num_p))
    bb_score = 1 - min(result_above_confidence_num_c,
                       result_above_confidence_num_p) / result_above_confidence_num_c

    return bb_score

def get_yolov4_bbscore(model, filepath):
    img_name = ntpath.basename(filepath)
    img_file_dir = 'ori_images'  
    img_path0 = os.path.join(img_file_dir, img_name)
    img0 = Image.open(img_path0).convert('RGB')
    img1 = Image.open(filepath).convert('RGB')

    resize_small = transforms.Compose([
        transforms.Resize((608, 608)),
    ])
    img0 = resize_small(img0)
    img1 = resize_small(img1)

    # --------------------BOX score
    boxes0 = do_detect(model, img0, 0.5, 0.4, True)
    boxes1 = do_detect(model, img1, 0.5, 0.4, True)

    assert len(boxes0) != 0
    bb_score = 1 - min(len(boxes0), len(boxes1))/len(boxes0)
    return bb_score

def get_cd_score(img_path, max_total_area_rate=0.02, max_patch_number=10):
    resize2 = transforms.Compose([
        transforms.ToTensor()])

    img_name = ntpath.basename(img_path)
    img_path0 = os.path.join('ori_images', img_name)
    img0 = Image.open(img_path0).convert('RGB')
    img1 = Image.open(img_path).convert('RGB')
    img0_t = resize2(img0).cuda()
    img1_t = resize2(img1).cuda()
    input_img = img0_t - img1_t

    ones = torch.cuda.FloatTensor(input_img[0].size()).fill_(1)
    zeros = torch.cuda.FloatTensor(input_img[0].size()).fill_(0)

    input_img_tmp2 = torch.where((input_img[0] != 0), ones, zeros) + \
                     torch.where((input_img[1] != 0), ones, zeros) + \
                     torch.where((input_img[2] != 0), ones, zeros)
    input_map_new = torch.where(input_img_tmp2 > 0, ones, zeros)

    whole_size = input_map_new.shape[0]*input_map_new.shape[1]

    labels = measure.label(input_map_new.cpu().numpy()[:, :], background=0, connectivity=2)
    label_max_number = np.max(labels)

    total_area = torch.sum(input_map_new).item()
    total_area_rate = total_area / whole_size
    
    area_score = 2 - float(total_area_rate/max_total_area_rate)
    connected_domin_score, total_area_rate, patch_number = float(area_score), float(total_area_rate), float(label_max_number)

    return not (patch_number > max_patch_number or patch_number == 0 or total_area_rate > max_total_area_rate), connected_domin_score

def get_score(image_path, model, darknet_model):
    valid, patch_score = get_cd_score(image_path)
    if(not valid):
        patch_score = 0
    score = patch_score * (get_yolov4_bbscore(darknet_model, image_path) + \
                               get_fasterrcnn_bbscore(model, image_path))
    return score

if __name__ == '__main__':
    MAX_TOTAL_AREA_RATE = 0.02  # 5000/(500*500) = 0.02
    candidate_dir = 'delete2'
    max_patch_number = 10

    config = '../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint = '../models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    model = init_detector(config, checkpoint, device='cuda:0')

    cfgfile = "../models/yolov4.cfg"
    weightfile = "../models/yolov4.weights"
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()

    #f2s = dict()
    #src = './merged/images'
    #for filename in tqdm(os.listdir(src)):
    #    if(not filename.endswith('.png')):
    #        continue
    #    f2s[filename] = get_score(os.path.join(src, filename), model, darknet_model)

    #with open('now_best.json', 'w') as f:
    #    json.dump(f2s, f)
    #exit(0)

    with open('now_best.json', 'r') as f:
        f2s = json.load(f)

    for filename in tqdm(os.listdir(candidate_dir)):
        if(not filename.endswith('.png')):
            continue

        img_path = os.path.join(candidate_dir, filename)
        score = get_score(img_path, model, darknet_model)
        if(score > f2s[filename]):
            print("%s: from %g to %g" % (filename, f2s[filename], score))
            f2s[filename] = score
            #copyfile(img_path, os.path.join('use', filename))

    total_score = 0
    for info in f2s.values():
        total_score += info
    print("Overall score = %g" % total_score)
    #with open('now_best.json', 'w') as f:
    #    json.dump(f2s, f)
