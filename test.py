from dataloader import DataTest
import torch
import os
import numpy as np
import random
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from models import ResNet, ResNetGRU, ResNetLSTM, ResNetConv1D


tried_once =  [2, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 32, 38, 40, 41, 42, 44, 47, 50, 51, 52, 53, 55,
    56, 57, 61, 63, 65, 67, 68, 70, 71, 72, 73, 74, 76, 80, 81, 83, 85, 87,
    88, 89, 90, 92, 93, 96, 97, 99, 101, 102, 105, 106, 108, 110, 111, 112,
    113, 114, 116, 118, 121, 123, 125, 131, 132, 134, 135, 140, 142, 143,
    145, 146, 148, 149, 151, 152, 154, 155, 156, 157, 160, 161, 162, 165,
    169, 170, 171, 173, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185,
    186, 187, 190, 191, 194, 196, 203, 204, 206, 207, 208, 209, 210, 211,
    213, 214, 216, 218, 219, 220, 222, 225, 227, 228, 229, 232, 233, 235,
    236, 237, 238, 239, 242, 243, 246, 247, 249, 251, 252, 254, 255, 256,
    257, 260, 261, 262, 263, 265, 266, 268, 270, 272, 275, 277, 278, 279,
    281, 282, 287, 290, 296, 298]
tried_twice = [1, 3, 11, 13, 14, 16, 17, 31, 33, 35, 36, 37, 39,
    43, 45, 49, 54, 58, 59, 60, 62, 64, 66, 69, 75, 77, 79, 82, 84, 86,
    91, 94, 95, 98, 100, 103, 104, 107, 109, 115, 117, 119, 120, 122,
    124, 126, 127, 128, 129, 130, 133, 136, 137, 138, 139, 141, 144,
    147, 150, 153, 158, 159, 164, 166, 167, 168, 172, 174, 177, 188,
    189, 192, 193, 195, 197, 198, 199, 200, 201, 202, 205, 212, 215,
    217, 221, 223, 224, 226, 230, 231, 234, 240, 241, 244, 245, 248,
    250, 253, 258, 259, 264, 267, 269, 271, 273, 274, 276, 280, 283,
    284, 285, 286, 288, 291, 292, 293, 294, 295, 297, 299]
tried_thrice = [34, 46, 48, 78, 163, 289]

friends = [i for i in range(150, 300)]
strangers = [i for i in range(0, 150)]


# test_frame_ids = tried_twice + tried_thrice
# test_frame_ids = tried_once
# test_frame_ids = friends
# test_frame_ids = strangers
test_frame_ids = None
# median = (240, False)
median = None

from model_resnet import Model
# from model_gru import Model
# from model_lstm import Model
# from model_conv1d import Model
model_type = 'resnet'
# model_type = 'gru'
# model_type = 'lstm'
# model_type = 'conv1d'
inp_dim = 512 # rgb
# inp_dim = 512 + 150 # pose
# inp_dim = 512 + 6 # gaze
# inp_dim = 512 + 108 # bbox
# inp_dim = 512 + 64 # ocr
# inp_dim = 512 + 108 + 64 # bbox + ocr
# inp_dim = 512 + 150 + 6 # pose + gaze
# inp_dim = 512 + 150 + 6 + 108 + 64 # all
load_model_path = '/path/to/model'
test_frame_path = '/path/to/test/frame'
test_pose_path = None # '/path/to/test/pose'
test_gaze_path = None # '/path/to/test/gaze'
test_bbox_path = None # '/path/to/test/bbox'
ocr_graph_path = None # '/path/to/OCRMap.txt'
label_path = '/path/to/label'
save_path = 'experiments/'
gpu_id = 3


def pad_collate(batch):
    (aa, bb, cc, dd, ee, ff) = zip(*batch)
    seq_lens = [len(a) for a in aa]
    aa_pad = pad_sequence(aa, batch_first=True, padding_value=0)
    bb_pad = pad_sequence(bb, batch_first=True, padding_value=0)
    if cc[0] is not None:
        cc_pad = pad_sequence(cc, batch_first=True, padding_value=0)
    else:
        cc_pad = None
    if dd[0] is not None:
        dd_pad = pad_sequence(dd, batch_first=True, padding_value=0)
    else:
        dd_pad = None
    if ee[0] is not None:
        ee_pad = pad_sequence(ee, batch_first=True, padding_value=0)
    else:
        ee_pad = None
    if ff[0] is not None:
        ff_pad = pad_sequence(ff, batch_first=True, padding_value=0)
    else:
        ff_pad = None
    return aa_pad, bb_pad, cc_pad, dd_pad, ee_pad, ff_pad, seq_lens #, y_lens


def get_classification_accuracy(pred_left_labels, pred_right_labels, labels, sequence_lengths):
    max_len = max(sequence_lengths)
    pred_left_labels = torch.reshape(pred_left_labels, (-1, max_len, 27))
    pred_right_labels = torch.reshape(pred_right_labels, (-1, max_len, 27))
    labels = torch.reshape(labels, (-1, max_len, 2))
    left_correct = torch.argmax(pred_left_labels, 2) == labels[:,:,0]
    right_correct = torch.argmax(pred_right_labels, 2) == labels[:,:,1]
    num_pred = sum(sequence_lengths) * 2
    num_correct = 0
    for i in range(len(sequence_lengths)):
        size = sequence_lengths[i]
        num_correct += (torch.sum(left_correct[i][:size]) + torch.sum(right_correct[i][:size])).item()
    acc = num_correct / num_pred

    return acc, num_correct, num_pred


def test():
    # get experiment ID
    experiment_id = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +  '_test'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    experiment_save_path = os.path.join(save_path, experiment_id)
    os.makedirs(experiment_save_path, exist_ok=True)
    
    # load datasets
    test_dataset = DataTest(test_frame_path, label_path, test_pose_path, test_gaze_path, test_bbox_path, ocr_graph_path, test_frame_ids, median)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=pad_collate)
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else 'cpu')
    assert load_model_path is not None
    if model_type == 'resnet':
        model = ResNet(inp_dim, device).to(device)
    elif model_type == 'gru':
        model = ResNetGRU(inp_dim, device).to(device)
    elif model_type == 'lstm':
        model = ResNetLSTM(inp_dim, device).to(device)
    elif model_type == 'conv1d':
        model = ResNetConv1D(inp_dim, device).to(device)
    model.load_state_dict(torch.load(load_model_path, map_location=device))
    model.device = device
    model.eval()

    print('Testing...')
    num_correct = 0
    cnt = 0
    with torch.no_grad():
        for j, batch in tqdm(enumerate(test_dataloader)):
            frames, labels, poses, gazes, bboxes, ocr_graphs, sequence_lengths = batch
            pred_left_labels, pred_right_labels = model(frames, poses, gazes, bboxes, ocr_graphs)
            pred_left_labels = torch.reshape(pred_left_labels, (-1, 27))
            pred_right_labels = torch.reshape(pred_right_labels, (-1, 27))
            labels = torch.reshape(labels, (-1, 2)).to(device)
            _, batch_num_correct, batch_num_pred = get_classification_accuracy(pred_left_labels, pred_right_labels, labels, sequence_lengths)
            cnt += batch_num_pred
            num_correct += batch_num_correct

    test_acc = num_correct / cnt
    print("Test accuracy: {}".format(num_correct / cnt))

    with open(os.path.join(experiment_save_path, 'test_stats.txt'), 'w') as f:
        f.write(str(test_acc))
        f.close()


if __name__ == '__main__':
    test()