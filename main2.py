from model_resnet_no_seq import Model
from dataloader import Data
from dataloader import Data2
import torch
import os
import numpy as np
import random
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


# args
train_frame_path = '/raid/data_yubjs/train/frame'
train_pose_path = None
train_gaze_path = None
train_bbox_path = None # '/raid/data_yubjs/train/bbox'
val_frame_path = '/raid/data_yubjs/val/frame'
val_pose_path = None
val_gaze_path = None
val_bbox_path = None # '/raid/data_yubjs/val/bbox'
ocr_graph_path = None # '/home/yubjs/MTOM/OCRMap.txt'
label_path = '/home/yubjs/MTOM/outfile'
load_model_path = None
batch_size = 4
num_epoch = 5
save_path = 'experiments_2dnoseq_OCR_Tmux11_fk'
gpu_id = 1


def pad_collate(batch):
    (aa, bb, cc, dd, ee, ff) = zip(*batch)
    seq_lens = [len(a) for a in aa]
    #   y_lens = [len(y) for y in yy]

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

    # NOTE: for no sequence scenario for 2D ResNet
    seq_lens = [1]

    return aa_pad, bb_pad, cc_pad, dd_pad, ee_pad, ff_pad, seq_lens #, y_lens


def get_classification_accuracy(pred_left_labels, pred_right_labels, labels, sequence_lengths):
    max_len = max(sequence_lengths)
    # print(sequence_lengths)
    pred_left_labels = torch.reshape(pred_left_labels, (-1, max_len, 27))
    pred_right_labels = torch.reshape(pred_right_labels, (-1, max_len, 27))
    labels = torch.reshape(labels, (-1, max_len, 2))
    left_correct = torch.argmax(pred_left_labels, 2) == labels[:,:,0]
    right_correct = torch.argmax(pred_right_labels, 2) == labels[:,:,1]
    num_pred = sum(sequence_lengths) * 2
    # num_correct = (torch.sum(left_correct[i][:size]) + torch.sum(right_correct[i][:size])).item()
    num_correct = 0
    for i in range(len(sequence_lengths)):
        size = sequence_lengths[i]
        num_correct += (torch.sum(left_correct[i][:size]) + torch.sum(right_correct[i][:size])).item()
    # print(pred_left_labels.shape, labels.shape)
    # num_correct = torch.sum(mask == labels).item()
    # acc = num_correct / size
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
    # train_dataset = Data(train_frame_path, label_path, train_pose_path, train_gaze_path, train_bbox_path, ocr_graph_path)
    train_dataset = Data2(train_frame_path, label_path, train_pose_path, train_gaze_path, train_bbox_path, ocr_graph_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=pad_collate)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    # val_dataset = Data(val_frame_path, label_path, val_pose_path, val_gaze_path, val_bbox_path, ocr_graph_path)
    val_dataset = Data2(val_frame_path, label_path, val_pose_path, val_gaze_path, val_bbox_path, ocr_graph_path)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=pad_collate)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else 'cpu')
    if load_model_path is not None:
        model = torch.load(load_model_path, map_location=device).to(device)
        model.device = device
    else:
        model = Model(device).to(device)
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
            pred_left_labels, pred_right_labels = model(frames, poses, gazes, bboxes, ocr_graphs)
            pred_left_labels = torch.reshape(pred_left_labels, (-1, 27))
            pred_right_labels = torch.reshape(pred_right_labels, (-1, 27))
            labels = torch.reshape(labels, (-1, 2)).to(device)
            # print(labels.shape, labels[:,0].shape, labels[:,1].shape, pred_left_labels.shape, pred_right_labels.shape)
            # calculate train performance metrics
            batch_train_acc, batch_num_correct, batch_num_pred = get_classification_accuracy(pred_left_labels, pred_right_labels, labels, sequence_lengths)
            epoch_cnt += batch_num_pred
            epoch_num_correct += batch_num_correct
            # print(pred_left_labels.shape, labels.shape)
            loss = cross_entropy_loss(pred_left_labels, labels[:,0]) + cross_entropy_loss(pred_right_labels, labels[:,1])
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
                # retrieved_batch_size = frames.shape[0]
                # total_cnt += retrieved_batch_size
                pred_left_labels, pred_right_labels = model(frames, poses, gazes, bboxes, ocr_graphs)
                pred_left_labels = torch.reshape(pred_left_labels, (-1, 27))
                pred_right_labels = torch.reshape(pred_right_labels, (-1, 27))
                labels = torch.reshape(labels, (-1, 2)).to(device)
                batch_val_acc, batch_num_correct, batch_num_pred = get_classification_accuracy(pred_left_labels, pred_right_labels, labels, sequence_lengths)
                # val_acc, num_correct = get_episode_classification_accuracy(pred_labels, labels)
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
            torch.save(model, os.path.join(experiment_save_path, 'model'))

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