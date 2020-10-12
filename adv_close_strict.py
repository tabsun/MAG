import torch
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.utils import *
from shutil import copyfile
import json
from sklearn.preprocessing import normalize
from scipy.ndimage.filters import gaussian_filter

import uuid
import random
import cv2
import mmcv
import numpy as np
import os
from tool.darknet2pytorch import *
from utils.utils import post_process
from infer import infer
from tqdm import tqdm
from skimage import measure

import sys
sys.path.append('./mmdetection/')
from mmdet import __version__
from mmdet.apis import init_detector, inference_detector

#def connect_regions_optimize(cd_map, labels):
def blur(grad):
    grad[0,0,:,:] = gaussian_filter(grad[0,0,:,:], sigma=3)
    grad[0,1,:,:] = gaussian_filter(grad[0,1,:,:], sigma=3)
    grad[0,2,:,:] = gaussian_filter(grad[0,2,:,:], sigma=3)
    return grad

def op(diff):
    diff = np.swapaxes(np.swapaxes(np.squeeze(diff), 0, 1), 1, 2)
    h, w = diff.shape[:2]
    diff = cv2.GaussianBlur(diff, (3,3), cv2.BORDER_DEFAULT)
    diff = np.swapaxes(np.swapaxes(diff, 1, 2), 0, 1).reshape(1,3,h,w)
    return diff

def check_image(diff):
    diff = np.swapaxes(np.swapaxes(np.squeeze(diff), 0, 1), 1, 2)
    h, w = diff.shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    diff_sum = np.squeeze(np.sum(np.abs(diff), axis=2))
    mask[:, :, 0] = 255 * (diff_sum > 0)
    mask[:, :, 1] = 255 * (diff_sum > 0) 
    mask[:, :, 2] = 255 * (diff_sum > 0) 
    # diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff)) * 255.
    return mask

def get_start_num(ori_img, diff_map, temp_fname, darknet_model, frcnn_model):
    save_format_try_image(ori_img, diff_map, temp_fname)
    yolo_input, frcnn_input = get_yolo_image(temp_fname), get_frcnn_image(temp_fname)
    yolo_input = torch.nn.functional.interpolate(yolo_input, size=(304,304), mode='bilinear')
    yolo_input = torch.nn.functional.interpolate(yolo_input, size=(608,608), mode='bilinear')
    frcnn_input = torch.nn.functional.interpolate(frcnn_input, size=(400,400), mode='bilinear')
    frcnn_input = torch.nn.functional.interpolate(frcnn_input, size=(800,800), mode='bilinear')

    list_boxes = darknet_model(yolo_input)
    yolo_results = post_process(list_boxes)
    yolo_num = len(yolo_results)

    frcnn_results, scores, _ = frcnn_model(img=[frcnn_input], img_metas=[[{'filename': '', 'ori_filename': '', 'ori_shape': (500, 500, 3), 'img_shape': (800, 800, 3), 'pad_shape': (800, 800, 3), 'scale_factor': np.array([1.6, 1.6, 1.6, 1.6]), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53 ]), 'std': np.array([58.395, 57.12 , 57.375]), 'to_rgb': True}}]], return_loss=False, rescale=False)
    frcnn_results = np.concatenate(frcnn_results) 
    frcnn_num = np.sum(frcnn_results[:, 4] > 0.3)
    return yolo_num, frcnn_num, yolo_results, frcnn_results

def update_one_model(ori_img, diff_map, temp_fname, best_temp_fname, darknet_model, frcnn_model, flag, start_yolo_num, start_frcnn_num, dest_num, rate, gt_bboxes, update_mask):
    print("Updating %s..." % flag)
    # generate bbox grad mask
    grad_mask = np.zeros((500,500,3), dtype=np.float)
    for bbox in gt_bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(grad_mask, (x1,y1), (x2,y2), (255,255,255), -1)
    grad_mask = np.swapaxes(np.swapaxes(grad_mask, 1, 2), 0, 1).reshape((1,3,500,500))

    step = 0
    max_steps_num = 200 if flag == 'frcnn' else 50
    best_yolo_num = start_yolo_num
    best_frcnn_num = start_frcnn_num
    min_yolo_loss = float('inf')
    min_frcnn_loss = float('inf')
    min_creterion = float('inf')
    best_diff_map = None
    gradient = np.zeros((1,3,500,500), dtype=np.float)

    relu = torch.nn.ReLU()
    while(step < max_steps_num):
        save_format_try_image(ori_img, diff_map, temp_fname)
        yolo_input, frcnn_input = get_yolo_image(temp_fname), get_frcnn_image(temp_fname)

        yolo_input.requires_grad = True
        list_boxes = darknet_model(yolo_input)
        yolo_results = post_process(list_boxes)
        yolo_num = len(yolo_results)

        boxes_0 = list_boxes[0].view(3, 85, -1)
        loss_0 = torch.sum(relu(boxes_0[:, 4, :]))
        boxes_1 = list_boxes[1].view(3, 85, -1)
        loss_1 = torch.sum(relu(boxes_1[:, 4, :]))
        boxes_2 = list_boxes[2].view(3, 85, -1)
        loss_2 = torch.sum(relu(boxes_2[:, 4, :]))
        yolo_loss = loss_0 + loss_1 + loss_2

        frcnn_input.requires_grad = True
        frcnn_results, scores, _ = frcnn_model(img=[frcnn_input], img_metas=[[{'filename': '', 'ori_filename': '', 'ori_shape': (500, 500, 3), 'img_shape': (800, 800, 3), 'pad_shape': (800, 800, 3), 'scale_factor': np.array([1.6, 1.6, 1.6, 1.6]), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53 ]), 'std': np.array([58.395, 57.12 , 57.375]), 'to_rgb': True}}]], return_loss=False, rescale=False)

        frcnn_results = np.concatenate(frcnn_results) 
        frcnn_loss  = torch.sum(relu(scores[:, :-1]-0.049))
        frcnn_num = np.sum(frcnn_results[:, 4] > 0.3)

        # # get gt bboxes
        # gt_bboxes = []
        # h = w = 500
        # for yolo_bbox in yolo_results:
        #     x1, y1, x2, y2 = yolo_bbox[:4]
        #     x1, x2 = int(x1*w), int(x2*w)
        #     y1, y2 = int(y1*h), int(y2*h)
        #     gt_bboxes.append([x1-x2//2, y1-y2//2, x1+x2//2, y1+y2//2])
        # for frcnn_bbox in frcnn_results:
        #     if(frcnn_bbox[-1] > 0.3):
        #         x1, y1, x2, y2 = [int(x/1.6) for x in frcnn_bbox[:4]]
        #         gt_bboxes.append([x1,y1,x2,y2])
        # # generate bbox grad mask
        # grad_mask = np.zeros((500,500,3), dtype=np.float)
        # for bbox in gt_bboxes:
        #     x1, y1, x2, y2 = bbox
        #     cv2.rectangle(grad_mask, (x1,y1), (x2,y2), (255,255,255), -1)
        # grad_mask = np.swapaxes(np.swapaxes(grad_mask, 1, 2), 0, 1).reshape((1,3,500,500))

        if(step == 0):
            epoch_creterion = float(yolo_num)/start_yolo_num + float(frcnn_num)/start_frcnn_num

        #creterion = yolo_num if flag == 'yolo' else frcnn_num
        creterion = 10000*(min(1., float(yolo_num) / start_yolo_num) + min(1., float(frcnn_num) / start_frcnn_num)) + (yolo_loss if flag == 'yolo' else frcnn_loss)
        if(creterion < min_creterion):
            min_creterion = creterion
            min_frcnn_loss = frcnn_loss
            min_yolo_loss = yolo_loss
            best_yolo_num = yolo_num
            best_frcnn_num = frcnn_num
            best_diff_map = diff_map.copy()
            copyfile(temp_fname, best_temp_fname)
        
        # check rate
        patch_number, area_rate = get_cd_score(fname, best_temp_fname)
        print("%d @ [%d,%d,  %d,%d  --> %d] f_loss=%g y_loss=%g min_f_loss=%g min_y_loss=%g, best patch=%d rate=%g limit=%.2f" % 
                    (step, yolo_num, frcnn_num, best_yolo_num, best_frcnn_num, dest_num,
                        frcnn_loss, yolo_loss, min_frcnn_loss, min_yolo_loss, patch_number, area_rate, 100.-rate))
        if(((yolo_num == 0 and flag == 'yolo') or (frcnn_num == 0 and flag == 'frcnn'))
                and area_rate < 0.02 and patch_number <= 10):
            break

        darknet_model.zero_grad()
        yolo_loss.backward(retain_graph=False)
        yolo_d_grad = yolo_input.grad.data.cpu().numpy().reshape((1,3,608,608))
        yolo_d_grad = np.swapaxes(np.swapaxes(yolo_d_grad[0], 0,1), 1,2)
        yolo_d_grad = mmcv.imresize(yolo_d_grad, (500,500))
        yolo_d_grad = np.swapaxes(np.swapaxes(yolo_d_grad, 1,2), 0,1).reshape((1,3,500,500))
        #yolo_d_grad = blur(yolo_d_grad)

        frcnn_model.zero_grad()
        frcnn_loss.backward(retain_graph=False)
        frcnn_d_grad = frcnn_input.grad.data.cpu().numpy().reshape((1,3,800,800))

        frcnn_d_grad[:, 0, :, :] = frcnn_d_grad[:, 0, :, :] *( 58.395/ 255.)
        frcnn_d_grad[:, 1, :, :] = frcnn_d_grad[:, 1, :, :] *( 57.12/ 255.)
        frcnn_d_grad[:, 2, :, :] = frcnn_d_grad[:, 2, :, :] *( 57.375/ 255.)
        frcnn_d_grad = np.swapaxes(np.swapaxes(frcnn_d_grad[0], 0,1), 1,2)
        frcnn_d_grad = mmcv.imresize(frcnn_d_grad, (500,500))
        frcnn_d_grad = np.swapaxes(np.swapaxes(frcnn_d_grad, 1,2), 0,1).reshape((1,3,500,500))
        #frcnn_d_norm = np.linalg.norm(frcnn_d_grad, ord=2, axis=1).reshape(500,500)
        #frcnn_d_norm = (frcnn_d_norm - np.min(frcnn_d_norm)) / (np.max(frcnn_d_norm) - np.min(frcnn_d_norm))
        #frcnn_weight = np.repeat(frcnn_d_norm.reshape(1,1,500,500), 3, axis=1)
        #frcnn_d_grad = np.multiply(frcnn_weight, frcnn_d_grad)
        frcnn_d_grad = normalize(frcnn_d_grad.reshape(3,-1), axis=1).reshape((1,3,500,500))
        frcnn_d_grad = frcnn_d_grad * 10
        #frcnn_d_grad = blur(frcnn_d_grad)

        if(flag == 'yolo'):
            alpha = 0.95
        else:
            alpha = 0.8
        gradient = (1.-alpha) * frcnn_d_grad + alpha * yolo_d_grad
        #if(flag == 'frcnn'):
        #    gradient = 0.9 * gradient + 0.1 * grad
        #else:
        #    gradient = grad

        loss = yolo_loss if flag == 'yolo' else frcnn_loss
        if(loss > 10):
            step_size = 2 #0.1 + 0.3*(float(loss)-10.)/(start_loss-10.)
        elif(loss > 5):
            step_size = 2
        else:
            step_size = 0.2
        step_size = step_size * (1. - float(step) / max_steps_num)

        gradient = step_size * gradient
        
        # blur
        #gradient[0,0,:,:] = gaussian_filter(gradient[0,0,:,:], sigma=3)
        #gradient[0,1,:,:] = gaussian_filter(gradient[0,1,:,:], sigma=3)
        #gradient[0,2,:,:] = gaussian_filter(gradient[0,2,:,:], sigma=3)

        # fix mask
        gradient *= update_mask.astype(np.float)

        diff_map -= gradient

        ## check area rate 
        #diff_map[grad_mask == 0] = 0
        #diff_map_change = np.sum(np.abs(diff_map), axis=1)
        #high_thresh = np.percentile(diff_map_change, rate)
        #gray_mask = ((diff_map_change > high_thresh) * 255.).astype(np.uint8)

        #gray_mask = gray_mask.reshape(500,500)
        #diff_map[0,0,:,:][gray_mask == 0] = 0
        #diff_map[0,1,:,:][gray_mask == 0] = 0
        #diff_map[0,2,:,:][gray_mask == 0] = 0

        ## check connected parts' number
        save_format_try_image(ori_img, diff_map, temp_fname)
        cd_map = get_cd_map(fname, temp_fname)
        labels = measure.label(cd_map, background=0, connectivity=2)
        label_num = np.max(labels)

        if(label_num > 10):
            areas = [np.sum(labels == i) for i in range(1, label_num+1)]
            label_ids = list(range(1, label_num+1))
            areas, label_ids = zip(*sorted(zip(areas, label_ids)))
            
            for i in label_ids[:-10]:
                #gray_mask[labels==i] = 0
                diff_map[0,0,:,:][labels==i] = 0
                diff_map[0,1,:,:][labels==i] = 0
                diff_map[0,2,:,:][labels==i] = 0

        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        #gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
        #gray_mask = gray_mask.reshape(500,500)
        #diff_map[0,0,:,:][gray_mask == 0] = 0
        #diff_map[0,1,:,:][gray_mask == 0] = 0
        #diff_map[0,2,:,:][gray_mask == 0] = 0
        #see = check_image(diff_map)
        #cv2.imwrite('check/%03d_region.jpg' % step, see)
        #cv2.imwrite('check/%03d_region_filter.jpg' % step, cv2.medianBlur(see, 3))
        step += 1

    return float(best_yolo_num)/start_yolo_num + float(best_frcnn_num)/start_frcnn_num >= epoch_creterion, best_diff_map
    #return [yolo_num, frcnn_num], best_diff_map

def get_cd_score(img_name, img_path, max_total_area_rate=0.02, max_patch_number=10):
    cd_map = get_cd_map(img_name, img_path)
    whole_size = cd_map.shape[0]*cd_map.shape[1]
    labels = measure.label(cd_map, background=0, connectivity=2)
    label_max_number = np.max(labels)
    total_area = np.sum(cd_map)
    total_area_rate = total_area / whole_size
    return label_max_number, total_area_rate

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

def get_binary_map(input_img):
    ones = torch.cuda.FloatTensor(input_img[0].size()).fill_(1)
    zeros = torch.cuda.FloatTensor(input_img[0].size()).fill_(0)

    input_img_tmp2 = torch.where((input_img[0] != 0), ones, zeros) + \
                     torch.where((input_img[1] != 0), ones, zeros) + \
                     torch.where((input_img[2] != 0), ones, zeros)
    input_map_new = torch.where(input_img_tmp2 > 0, ones, zeros)
    return input_map_new

def get_rate(diff_map):
    return np.sum(np.sum(diff_map, axis=1) != 0) / (500.*500.)

def percentile(t, q):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result

def get_frcnn_image(temp_fname):
    image = mmcv.imread(temp_fname)
    image = mmcv.imresize(image, (800,800))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(image)
    img = img.view(800, 800, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3, 800, 800).float()
    img[:, 0, :, :] = (img[:, 0, :, :] - 123.675) / 58.395
    img[:, 1, :, :] = (img[:, 1, :, :] - 116.28) / 57.12
    img[:, 2, :, :] = (img[:, 2, :, :] - 103.53) / 57.375 
    img = img.cuda()
    return img

def get_yolo_image(temp_fname):
    try_image = Image.open(temp_fname)
    resize_small = transforms.Compose([
        transforms.Resize((608, 608)),
    ])
    try_image = resize_small(try_image)
    width = try_image.width
    height = try_image.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(try_image.tobytes()))
    img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0).view((1,3,608,608)).cuda()
    return img

def save_format_try_image(ori_img, diff_map, temp_fname):
    try_image = np.clip(np.around((ori_img+diff_map)*255.), 0, 255).astype(np.uint8)
    try_image = np.swapaxes(np.swapaxes(try_image[0], 0, 1), 1, 2)
    try_image = Image.fromarray(try_image)
    try_image.save(temp_fname)
    return

def frcnn_preprocess(img):
    # mean = [123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    #img[:, 0, :, :] = (img[:, 0, :, :] - 123.675) / 58.395
    #img[:, 1, :, :] = (img[:, 1, :, :] - 116.28) / 57.12 
    #img[:, 2, :, :] = (img[:, 2, :, :] - 103.53) / 57.375
    img = img[:, [2,1,0], :, :]
    img = torch.nn.functional.interpolate(img, size=(800, 800), mode='bilinear', align_corners=False)
    return img

def yolo_preprocess(img):
    #img = img.float().div(255.0)
    #img = torch.clamp(img[:, [2,1,0], :, :], 0, 1)
    img = torch.nn.functional.interpolate(img, size=(608, 608), mode='bilinear', align_corners=False)
    return img

if __name__ == '__main__':
    # yolov4 model
    cfgfile = "models/yolov4.cfg"
    weightfile = "models/yolov4.weights"
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()
    # faster rcnn model
    config = './mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint = './models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    frcnn_model = init_detector(config, checkpoint, device='cuda:0')
    
    # cascade rcnn model
    #config = './mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
    #checkpoint = './models/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
    #crcnn_model = init_detector(config, checkpoint, device='cuda:0')

    dst_dir = './select1000_new_p'
    root = './select1000_new'
    fnames = [x for x in os.listdir(root)]
    random.shuffle(fnames)
    #with open('submit/merged/v1_v9_best.json', 'r') as f:
    #    data = json.load(f)
    data = dict()
    
    with open('./best_output_data/connected_domin_score.json', 'r') as f:
        cd_scores = json.load(f)
    with open('./best_output_data/whitebox_fasterrcnn_boundingbox_score.json', 'r') as f:
        f_scores = json.load(f)
    with open('./best_output_data/whitebox_yolo_boundingbox_score.json', 'r') as f:
        y_scores = json.load(f)

    #with open('submit/valid.txt', 'r') as f:
    #    valid_fnames = [line.strip() for line in f.readlines()]

    for fname in tqdm(fnames):
        if(fname in data and data[fname][0] > 0):
            continue
        #if(fname in valid_fnames):
        #    continue
        
        print(fname)
        if(not fname.endswith('png') or os.path.exists(os.path.join(dst_dir, fname))):
            continue
        temp_fname = os.path.join('temp', fname+'_temp_'+uuid.uuid4().hex[:6].upper()+'.png')
        best_temp_fname = temp_fname + '_best.png'
        image_path = os.path.join(root, fname) 
        # DEBUG
        update_mask = np.load('masks/%s.npy'%fname)

        img = Image.open(image_path).convert('RGB')
        height = 500
        width = 500
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width).float().cuda()
        yolo_img = img / 255.
        ori_img = yolo_img.cpu().numpy()
        
        best_diff_map = np.zeros((1,3,500,500), dtype=np.float32)
        min_frcnn_loss = float('inf')
        min_yolo_loss = float('inf')
        start_yolo_num, start_frcnn_num, yolo_bboxes, frcnn_bboxes = get_start_num(ori_img, best_diff_map, temp_fname, darknet_model, frcnn_model)

        gt_bboxes = []
        # DEBUG
        #h, w = debug_image.shape[:2]
        h = w = 500
        for yolo_bbox in yolo_bboxes:
            x1, y1, x2, y2 = yolo_bbox[:4]
            x1, x2 = int(x1*w), int(x2*w)
            y1, y2 = int(y1*h), int(y2*h)
            gt_bboxes.append([x1-x2//2, y1-y2//2, x1+x2//2, y1+y2//2])
        for frcnn_bbox in frcnn_bboxes:
            if(frcnn_bbox[-1] > 0.3):
                x1, y1, x2, y2 = [int(x/1.6) for x in frcnn_bbox[:4]]
                gt_bboxes.append([x1,y1,x2,y2])

        start_yolo_num, start_frcnn_num = max(1, start_yolo_num), max(1, start_frcnn_num)
        dest_yolo_num = int(round((1. - y_scores[fname]) * start_yolo_num))
        dest_frcnn_num = int(round((1. - f_scores[fname]) * start_frcnn_num))
        rate = 100. - 2*(2. - cd_scores[fname])

        flag_id = 0
        flags = ['frcnn', 'yolo']
        destinations = [False, False]
        while(True):
            flag = flags[flag_id]
            dest_num = dest_yolo_num if flag == 'yolo' else dest_frcnn_num
            dest, best_diff_map = update_one_model(ori_img, best_diff_map, temp_fname, best_temp_fname, darknet_model, frcnn_model, flag, start_yolo_num, start_frcnn_num, dest_num, rate, gt_bboxes, update_mask)
            if(not dest):
                destinations = [False, False]
            else:
                destinations[flag_id] = True
            print(destinations)

            #if(cur_nums[0] <= dest_yolo_num and cur_nums[1] <= dest_frcnn_num):
            if(destinations[0] and destinations[1]):
                break
            flag_id = (flag_id + 1) % 2
            
        patch_num, area_rate = get_cd_score(fname, best_temp_fname)
        print("Final image check connected parts %d, %g" % (patch_num, area_rate))
        os.rename(best_temp_fname, os.path.join(dst_dir, fname))
        os.remove(temp_fname)
