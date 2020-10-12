import torch
from torchvision import transforms
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.utils import *
import random
import json
import mmcv
import cv2

import uuid
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

def get_fasterrcnn_num(model, filepath, gt_bboxes=None):
    show_score_thr = 0.3
    file_name = ntpath.basename(filepath)

    result_p, _, rpn_loss = inference_detector(model, filepath, gt_bboxes=gt_bboxes)
    if isinstance(result_p, tuple):
        bbox_results, _ = result_p
        result_p = bbox_results
    result_above_confidence_num_p = 0
    result_p = np.concatenate(result_p)
    
    for ir in range(len(result_p)):
        if result_p[ir, 4] > show_score_thr:
            result_above_confidence_num_p = result_above_confidence_num_p + 1
    return result_above_confidence_num_p

def get_yolov4_num(model, filepath):
    img1 = Image.open(filepath).convert('RGB')

    resize_small = transforms.Compose([
        transforms.Resize((608, 608)),
    ])
    img1 = resize_small(img1)

    # --------------------BOX score
    boxes1 = do_detect(model, img1, 0.5, 0.4, True)

    return len(boxes1)

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

def get_cd_map(img_name, img_path, max_total_area_rate=0.02, max_patch_number=10):
    resize2 = transforms.Compose([
        transforms.ToTensor()])

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

    return input_map_new.cpu().numpy()[:,:]


if __name__ == '__main__':
    MAX_TOTAL_AREA_RATE = 0.02  # 5000/(500*500) = 0.02
    max_patch_number = 10

    config = '../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint = '../models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    model = init_detector(config, checkpoint, device='cuda:0')

    cfgfile = "../models/yolov4.cfg"
    weightfile = "../models/yolov4.weights"
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()

    root = 'merged/images'
    filenames = []
    for filename in os.listdir(root):
        if(filename.endswith('.png') and not os.path.exists(os.path.join('best_images', filename))):
            filenames.append(filename)
    random.shuffle(filenames)
    print("Total %d images" % len(filenames))

    for filename in filenames:
        if(os.path.exists(os.path.join('delete', filename))):
            continue
        img_path = os.path.join(root, filename)
        yolo_num, frcnn_num = get_yolov4_num(darknet_model, img_path), get_fasterrcnn_num(model, img_path)
        #yolo_num = get_yolov4_num(darknet_model, img_path)
        image = cv2.imread(img_path)
        ori_image = cv2.imread(os.path.join('ori_images', filename))
        temp_fname = os.path.join('temp', filename+'_'+uuid.uuid4().hex[:6].upper()+'.png')
        cd_map = get_cd_map(filename, img_path)
        pt_num = np.sum(cd_map != 0)

        count = 0
        update = False
        for y in tqdm(range(500)):
            for x in range(500):
                if(cd_map[y,x] > 0):
                    try_map = cd_map.copy()
                    try_map[y, x] = 0
                    labels = measure.label(try_map, background=0, connectivity=2)
                    label_max_number = np.max(labels)
                    if(label_max_number <= 10):
                        try_image = image.copy()
                        try_image[y,x,:] = ori_image[y,x,:]
                        cv2.imwrite(temp_fname, try_image)
                        frcn_flag = get_fasterrcnn_num(model, temp_fname) <= frcnn_num
                        if(frcn_flag):
                            yolo_flag = get_yolov4_num(darknet_model, temp_fname) <= yolo_num
                            if(yolo_flag):
                                cd_map[y,x] = 0
                                image[y,x,:] = ori_image[y,x,:]
                                count += 1
                                update = True
        print("Delete %d in %d" % (count, pt_num))
        os.remove(temp_fname)
        cv2.imwrite(os.path.join('delete', filename), image)

