from dataloader import Data
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


# args
batch_size = 2
num_epoch = 5
# model_type = 'resnet'
# model_type = 'gru'
model_type = 'lstm'
# model_type = 'conv1d'
# inp_dim = 512 # rgb
# inp_dim = 512 + 150 # pose
# inp_dim = 512 + 6 # gaze
# inp_dim = 512 + 108 # bbox
# inp_dim = 512 + 64 # ocr
# inp_dim = 512 + 108 + 64 # bbox + ocr
# inp_dim = 512 + 150 + 6 # pose + gaze
inp_dim = 512 + 150 + 6 + 108 + 64 # all
train_frame_path = '/mnt/c/Users/samso/Desktop/test/test/frame'
train_pose_path = '/mnt/c/Users/samso/Desktop/test/test/pose'
train_gaze_path = '/mnt/c/Users/samso/Desktop/test/test/gaze'
train_bbox_path = '/mnt/c/Users/samso/Desktop/test/test/bbox'
val_frame_path = '/mnt/c/Users/samso/Desktop/test/test/frame'
val_pose_path = '/mnt/c/Users/samso/Desktop/test/test/pose'
val_gaze_path = '/mnt/c/Users/samso/Desktop/test/test/gaze'
val_bbox_path = '/mnt/c/Users/samso/Desktop/test/test/bbox'
ocr_graph_path = '/mnt/c/Users/samso/Desktop/test/test/OCRMap.txt'
label_path = '/mnt/c/Users/samso/Desktop/test/test/outfile'
save_path = '/mnt/c/Users/samso/Desktop/test/test/experiments/'
gpu_id = 7


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


def train():
    # get experiment ID
    experiment_id = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') +  '_train'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    experiment_save_path = os.path.join(save_path, experiment_id)
    os.makedirs(experiment_save_path, exist_ok=True)
    
    # load datasets
    train_dataset = Data(train_frame_path, label_path, train_pose_path, train_gaze_path, train_bbox_path, ocr_graph_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=pad_collate)
    val_dataset = Data(val_frame_path, label_path, val_pose_path, val_gaze_path, val_bbox_path, ocr_graph_path)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=pad_collate)
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else 'cpu')
    if model_type == 'resnet':
        model = ResNet(inp_dim, device).to(device)
    elif model_type == 'gru':
        model = ResNetGRU(inp_dim, device).to(device)
    elif model_type == 'lstm':
        model = ResNetLSTM(inp_dim, device).to(device)
    elif model_type == 'conv1d':
        model = ResNetConv1D(inp_dim, device).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    cross_entropy_loss = nn.CrossEntropyLoss().to(device)
    stats = {'train': {'cls_loss': [], 'cls_acc': []}, 'val': {'cls_loss': [], 'cls_acc': []}}
    
    max_val_classification_acc = 0
    max_val_classification_epoch = None

    for i in range(num_epoch):
        # training
        print('Training for epoch {}/{}...'.format(i+1, num_epoch))
        temp_train_classification_loss = []
        epoch_num_correct = 0
        epoch_cnt = 0
        model.train()
        for j, batch in tqdm(enumerate(train_dataloader)):
            frames, labels, poses, gazes, bboxes, ocr_graphs, sequence_lengths = batch
            pred_left_labels, pred_right_labels, mm_loss = model(frames, poses, gazes, bboxes, ocr_graphs, labels)
            pred_left_labels = torch.reshape(pred_left_labels, (-1, 27))
            pred_right_labels = torch.reshape(pred_right_labels, (-1, 27))
            labels = torch.reshape(labels, (-1, 2)).to(device)
            batch_train_acc, batch_num_correct, batch_num_pred = get_classification_accuracy(pred_left_labels, pred_right_labels, labels, sequence_lengths)
            epoch_cnt += batch_num_pred
            epoch_num_correct += batch_num_correct
            # merge losses in a way that has similar magnitude
            loss = cross_entropy_loss(pred_left_labels, labels[:,0]) + cross_entropy_loss(pred_right_labels, labels[:,1]) + mm_loss
            temp_train_classification_loss.append(loss.data.item() * batch_num_pred / 2)

            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch {}/{} batch {}/{} training done with cls loss={}, cls accuracy={}.".format(i+1, num_epoch, j+1, len(train_dataloader), loss.data.item(), batch_train_acc))
            # break

        print("Epoch {}/{} OVERALL train cls loss={}, cls accuracy={}.\n".format(i+1, num_epoch, sum(temp_train_classification_loss) * 2 / epoch_cnt, epoch_num_correct / epoch_cnt))
        stats['train']['cls_loss'].append(sum(temp_train_classification_loss) * 2 / epoch_cnt)
        stats['train']['cls_acc'].append(epoch_num_correct / epoch_cnt)

        # validation
        print('Validation for epoch {}/{}...'.format(i+1, num_epoch))
        temp_val_classification_loss = []
        epoch_num_correct = 0
        epoch_cnt = 0
        model.eval()
        with torch.no_grad():
            for j, batch in tqdm(enumerate(val_dataloader)):
                frames, labels, poses, gazes, bboxes, ocr_graphs, sequence_lengths = batch
                pred_left_labels, pred_right_labels, mm_loss = model(frames, poses, gazes, bboxes, ocr_graphs, labels)
                pred_left_labels = torch.reshape(pred_left_labels, (-1, 27))
                pred_right_labels = torch.reshape(pred_right_labels, (-1, 27))
                labels = torch.reshape(labels, (-1, 2)).to(device)
                batch_val_acc, batch_num_correct, batch_num_pred = get_classification_accuracy(pred_left_labels, pred_right_labels, labels, sequence_lengths)
                epoch_cnt += batch_num_pred
                epoch_num_correct += batch_num_correct
                loss = cross_entropy_loss(pred_left_labels, labels[:,0]) + cross_entropy_loss(pred_right_labels, labels[:,1])
                temp_val_classification_loss.append(loss.data.item() * batch_num_pred / 2)

                print("Epoch {}/{} batch {}/{} validation done with cls loss={}, cls accuracy={}.".format(i+1, num_epoch, j+1, len(val_dataloader), loss.data.item(), batch_val_acc))

        print("Epoch {}/{} OVERALL validation cls loss={}, cls accuracy={}.\n".format(i+1, num_epoch, sum(temp_val_classification_loss) * 2 / epoch_cnt, epoch_num_correct / epoch_cnt))
        stats['val']['cls_loss'].append(sum(temp_val_classification_loss) * 2 / epoch_cnt)
        stats['val']['cls_acc'].append(epoch_num_correct / epoch_cnt)

        # check for best stat/model using validation accuracy
        if stats['val']['cls_acc'][-1] >= max_val_classification_acc:
            max_val_classification_acc = stats['val']['cls_acc'][-1]
            max_val_classification_epoch = i+1
            torch.save(model.state_dict(), os.path.join(experiment_save_path, 'model'))

        with open(os.path.join(experiment_save_path, 'log.txt'), 'w') as f:
            # f.write('{}\n'.format(cfg))
            f.write('{}\n'.format(stats))
            f.write('Max val classification acc: epoch {}, {}\n'.format(max_val_classification_epoch, max_val_classification_acc))
            f.close()



if __name__ == '__main__':
    # set seed values
    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train()