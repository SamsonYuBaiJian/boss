from tkinter import E
from torch.utils.data import Dataset
import os
from skimage import io
from torchvision import transforms
from natsort import natsorted
import numpy as np
import pickle
import torch
import cv2
import ast


class Data(Dataset):
    def __init__(self, frame_path, label_path, pose_path=None, gaze_path=None, bbox_path=None, ocr_graph_path=None):
        self.data = {'frame_paths': [], 'labels': [], 'poses': [], 'gazes': [], 'bboxes': [], 'ocr_graph': []}
        self.size_1, self.size_2 = 128, 128
        frame_dirs = os.listdir(frame_path)
        episode_ids = natsorted(frame_dirs)
        seq_lens = []
        for frame_dir in natsorted(frame_dirs):
            frame_paths = [os.path.join(frame_path, frame_dir, i) for i in natsorted(os.listdir(os.path.join(frame_path, frame_dir)))]
            seq_lens.append(len(frame_paths))
            self.data['frame_paths'].append(frame_paths)

        print(sorted(seq_lens)[89:91], len(seq_lens))

        with open(label_path, 'rb') as fp:
            labels = pickle.load(fp)
            left_labels, right_labels = labels
            for i in episode_ids:
                episode_id = int(i)
                left_labels_i = left_labels[episode_id]
                right_labels_i = right_labels[episode_id]
                # right_labels = pickle.load(fp)
                self.data['labels'].append([left_labels_i, right_labels_i])
            fp.close()

        self.pose_path = pose_path
        if pose_path is not None:
            for i in episode_ids:
                pose_i_dir = os.path.join(pose_path, i)
                poses = []
                for j in natsorted(os.listdir(pose_i_dir)):
                    fs = cv2.FileStorage(os.path.join(pose_i_dir, j), cv2.FILE_STORAGE_READ)
                    if torch.tensor(fs.getNode("pose_0").mat()).shape != (2,25,3):
                        poses.append(fs.getNode("pose_0").mat()[:2,:,:])
                    else:
                        poses.append(fs.getNode("pose_0").mat())
                self.data['poses'].append(torch.flatten(torch.tensor(poses), 1))

        self.gaze_path = gaze_path
        if gaze_path is not None:
            for i in episode_ids:
                gaze_i_txt = os.path.join(gaze_path, '{}.txt'.format(i))
                with open(gaze_i_txt, 'r') as fp:
                    gaze_i = fp.readlines()
                    fp.close()
                gaze_i = ast.literal_eval(gaze_i[0])
                gaze_i_tensor = torch.zeros((len(gaze_i), 2, 3))
                for j in range(len(gaze_i)):
                    if len(gaze_i[j]) >= 2:
                        gaze_i_tensor[j,:] = torch.tensor(gaze_i[j][:2])
                    elif len(gaze_i[j]) == 1:
                        gaze_i_tensor[j,0] = torch.tensor(gaze_i[j][0])
                    else:
                        continue
                self.data['gazes'].append(torch.flatten(gaze_i_tensor, 1))

        self.bbox_path = bbox_path
        if bbox_path is not None:
            self.objects = [
                'none', 'apple', 'orange', 'lemon', 'potato', 'wine', 'wineopener',
                'knife', 'mug', 'peeler', 'bowl', 'chocolate', 'sugar', 'magazine',
                'cracker', 'chips', 'scissors', 'cap', 'marker', 'sardinecan', 'tomatocan',
                'plant', 'walnut', 'nail', 'waterspray', 'hammer', 'canopener'
            ]
            for i in episode_ids:
                bbox_i_dir = os.path.join(bbox_path, i)
                with open(bbox_i_dir, 'rb') as fp:
                    bboxes_i = pickle.load(fp)
                    len_i = len(bboxes_i)
                    fp.close()
                bboxes_i_tensor = torch.zeros((len_i, len(self.objects), 4))
                for j in range(len(bboxes_i)):
                    items_i_j, bboxes_i_j = bboxes_i[j]
                    for k in range(len(items_i_j)):
                        bboxes_i_tensor[j, self.objects.index(items_i_j[k])] = torch.tensor([
                            bboxes_i_j[k][0] / 1920 * self.size_1,
                            bboxes_i_j[k][1] / 1088 * self.size_2,
                            bboxes_i_j[k][2] / 1920 * self.size_1,
                            bboxes_i_j[k][3] / 1088 * self.size_2
                        ]) # [x_min, y_min, x_max, y_max]
                self.data['bboxes'].append(torch.flatten(bboxes_i_tensor, 1))

        self.ocr_graph_path = ocr_graph_path
        if ocr_graph_path is not None:
            # with open(ocr_graph_path, 'r') as f:
            #     ocr_graph = f.readlines()[0]
            #     f.close()
            ocr_graph = [[15, [10, 4], [17, 2]], [13, [16, 7], [18, 4]], [11, [16, 4], [7, 10]], [14, [10, 11], [7, 1]], [12, [10, 9], [16, 3]], [1, [
                7, 2], [9, 9]], [5, [8, 8], [6, 8]], [4, [9, 8], [7, 6]], [3, [10, 1], [8, 3]], [2, [10, 1], [7, 7]], [19, [10, 2], [26, 6]], 
                [20, [10, 7], [26, 5]], [22, [25, 4], [10, 8]], [24, [10, 11], [7, 1]], [23, [16, 4], [7, 10]]]
            ocr_tensor = torch.zeros((27, 27))
            for ocr in ocr_graph:
                obj = ocr[0]
                contexts = ocr[1:]
                total_context_count = sum([i[1] for i in contexts])
                for context in contexts:
                    ocr_tensor[obj, context[0]] = context[1] / total_context_count
            ocr_tensor = torch.flatten(ocr_tensor)
            for i in episode_ids:
                self.data['ocr_graph'].append(ocr_tensor)

        self.frame_path = frame_path
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.size_1, self.size_2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.data['frame_paths'])

    def __getitem__(self, idx):
        frame_paths = self.data['frame_paths'][idx]
        images = torch.stack([self.transform(io.imread(i)) for i in frame_paths])

        if self.pose_path is not None:
            pose_paths = self.data['poses'][idx]
        else:
            pose_paths = None

        if self.gaze_path is not None:
            gaze_paths = self.data['gazes'][idx]
        else:
            gaze_paths = None

        if self.bbox_path is not None:
            bbox_paths = self.data['bboxes'][idx]
        else:
            bbox_paths = None

        if self.ocr_graph_path is not None:
            ocr_graphs = self.data['ocr_graph'][idx]
        else:
            ocr_graphs = None

        return images, torch.permute(torch.tensor(self.data['labels'][idx]), (1, 0)), pose_paths, gaze_paths, bbox_paths, ocr_graphs


class DataTest(Dataset):
    def __init__(self, frame_path, label_path, pose_path=None, gaze_path=None, bbox_path=None, ocr_graph_path=None, frame_ids=None, median=None):
        self.data = {'frame_paths': [], 'labels': [], 'poses': [], 'gazes': [], 'bboxes': [], 'ocr_graph': []}
        self.size_1, self.size_2 = 128, 128
        assert not (frame_ids is not None and median_len is not None), "Choose only one from frame_ids or median_len."
        if frame_ids is not None:
            test_ids = []
            frame_dirs = os.listdir(frame_path)
            episode_ids = natsorted(frame_dirs)
            for frame_dir in episode_ids:
                if int(frame_dir) in frame_ids:
                    test_ids.append(int(frame_dir))
                    frame_paths = [os.path.join(frame_path, frame_dir, i) for i in natsorted(os.listdir(os.path.join(frame_path, frame_dir)))]
                    self.data['frame_paths'].append(frame_paths)
        elif median is not None:
            test_ids = []
            frame_dirs = os.listdir(frame_path)
            episode_ids = natsorted(frame_dirs)
            for frame_dir in episode_ids:
                frame_paths = [os.path.join(frame_path, frame_dir, i) for i in natsorted(os.listdir(os.path.join(frame_path, frame_dir)))]
                seq_len = len(frame_paths)
                if (median[1] and seq_len >= median[0]) or (not median[1] and seq_len < median[0]):
                    self.data['frame_paths'].append(frame_paths)
                    test_ids.append(int(frame_dir))
        else:
            frame_dirs = os.listdir(frame_path)
            episode_ids = natsorted(frame_dirs)
            test_ids = episode_ids.copy()
            for frame_dir in episode_ids:
                frame_paths = [os.path.join(frame_path, frame_dir, i) for i in natsorted(os.listdir(os.path.join(frame_path, frame_dir)))]
                self.data['frame_paths'].append(frame_paths)

        with open(label_path, 'rb') as fp:
            labels = pickle.load(fp)
            left_labels, right_labels = labels
            for i in test_ids:
                episode_id = int(i)
                left_labels_i = left_labels[episode_id]
                right_labels_i = right_labels[episode_id]
                self.data['labels'].append([left_labels_i, right_labels_i])
            fp.close()

        self.pose_path = pose_path
        if pose_path is not None:
            for i in test_ids:
                pose_i_dir = os.path.join(pose_path, i)
                poses = []
                for j in natsorted(os.listdir(pose_i_dir)):
                    fs = cv2.FileStorage(os.path.join(pose_i_dir, j), cv2.FILE_STORAGE_READ)
                    if torch.tensor(fs.getNode("pose_0").mat()).shape != (2,25,3):
                        poses.append(fs.getNode("pose_0").mat()[:2,:,:])
                    else:
                        poses.append(fs.getNode("pose_0").mat())
                self.data['poses'].append(torch.flatten(torch.tensor(poses), 1))

        self.gaze_path = gaze_path
        if gaze_path is not None:
            for i in episode_ids:
                gaze_i_txt = os.path.join(gaze_path, '{}.txt'.format(i))
                with open(gaze_i_txt, 'r') as fp:
                    gaze_i = fp.readlines()
                    fp.close()
                gaze_i = ast.literal_eval(gaze_i[0])
                gaze_i_tensor = torch.zeros((len(gaze_i), 2, 3))
                for j in range(len(gaze_i)):
                    if len(gaze_i[j]) >= 2:
                        gaze_i_tensor[j,:] = torch.tensor(gaze_i[j][:2])
                    elif len(gaze_i[j]) == 1:
                        gaze_i_tensor[j,0] = torch.tensor(gaze_i[j][0])
                    else:
                        continue
                self.data['gazes'].append(torch.flatten(gaze_i_tensor, 1))

        self.bbox_path = bbox_path
        if bbox_path is not None:
            self.objects = [
                'none', 'apple', 'orange', 'lemon', 'potato', 'wine', 'wineopener',
                'knife', 'mug', 'peeler', 'bowl', 'chocolate', 'sugar', 'magazine',
                'cracker', 'chips', 'scissors', 'cap', 'marker', 'sardinecan', 'tomatocan',
                'plant', 'walnut', 'nail', 'waterspray', 'hammer', 'canopener'
            ]
            for i in test_ids:
                bbox_i_dir = os.path.join(bbox_path, i)
                with open(bbox_i_dir, 'rb') as fp:
                    bboxes_i = pickle.load(fp)
                    len_i = len(bboxes_i)
                    fp.close()
                bboxes_i_tensor = torch.zeros((len_i, len(self.objects), 4))
                for j in range(len(bboxes_i)):
                    items_i_j, bboxes_i_j = bboxes_i[j]
                    for k in range(len(items_i_j)):
                        bboxes_i_tensor[j, self.objects.index(items_i_j[k])] = torch.tensor([
                            bboxes_i_j[k][0] / 1920 * self.size_1,
                            bboxes_i_j[k][1] / 1088 * self.size_2,
                            bboxes_i_j[k][2] / 1920 * self.size_1,
                            bboxes_i_j[k][3] / 1088 * self.size_2
                        ]) # [x_min, y_min, x_max, y_max]
                self.data['bboxes'].append(torch.flatten(bboxes_i_tensor, 1))

        self.ocr_graph_path = ocr_graph_path
        if ocr_graph_path is not None:
            # with open(ocr_graph_path, 'r') as f:
            #     ocr_graph = f.readlines()[0]
            #     f.close()
            # print(ocr_graph)
            ocr_graph = [[15, [10, 4], [17, 2]], [13, [16, 7], [18, 4]], [11, [16, 4], [7, 10]], [14, [10, 11], [7, 1]], [12, [10, 9], [16, 3]], [1, [
                7, 2], [9, 9]], [5, [8, 8], [6, 8]], [4, [9, 8], [7, 6]], [3, [10, 1], [8, 3]], [2, [10, 1], [7, 7]], [19, [10, 2], [26, 6]], 
                [20, [10, 7], [26, 5]], [22, [25, 4], [10, 8]], [24, [10, 11], [7, 1]], [23, [16, 4], [7, 10]]]
            ocr_tensor = torch.zeros((27, 27))
            for ocr in ocr_graph:
                obj = ocr[0]
                contexts = ocr[1:]
                total_context_count = sum([i[1] for i in contexts])
                for context in contexts:
                    ocr_tensor[obj, context[0]] = context[1] / total_context_count
            ocr_tensor = torch.flatten(ocr_tensor)
            for i in test_ids:
                self.data['ocr_graph'].append(ocr_tensor)

        self.frame_path = frame_path
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.size_1, self.size_2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.data['frame_paths'])

    def __getitem__(self, idx):
        frame_paths = self.data['frame_paths'][idx]
        images = torch.stack([self.transform(io.imread(i)) for i in frame_paths])

        if self.pose_path is not None:
            pose_paths = self.data['poses'][idx]
        else:
            pose_paths = None

        if self.gaze_path is not None:
            gaze_paths = self.data['gazes'][idx]
        else:
            gaze_paths = None

        if self.bbox_path is not None:
            bbox_paths = self.data['bboxes'][idx]
        else:
            bbox_paths = None

        if self.ocr_graph_path is not None:
            ocr_graphs = self.data['ocr_graph'][idx]
        else:
            ocr_graphs = None

        return images, torch.permute(torch.tensor(self.data['labels'][idx]), (1, 0)), pose_paths, gaze_paths, bbox_paths, ocr_graphs