import torch
import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, device):
        super(ResNet, self).__init__()
        # Conv
        resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))
        # FFs
        inp_dim = 512 # rgb
        # inp_dim = 512 + 150 # pose
        # inp_dim = 512 + 6 # gaze
        # inp_dim = 512 + 108 # bbox
        # inp_dim = 512 + 64 # ocr
        # inp_dim = 512 + 108 + 64 # bbox + ocr
        # inp_dim = 512 + 150 + 6 # pose + gaze
        # inp_dim = 512 + 150 + 6 + 108 + 64 # all
        self.left = nn.Linear(inp_dim, 27)
        self.right = nn.Linear(inp_dim, 27)
        # modality FFs
        self.pose_ff = nn.Linear(150, 150)
        self.gaze_ff = nn.Linear(6, 6)
        self.bbox_ff = nn.Linear(108, 108)
        self.ocr_ff = nn.Linear(729, 64)
        # others
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.device = device

    def forward(self, images, poses, gazes, bboxes, ocr_tensor):
        batch_size, sequence_len, channels, height, width = images.shape
        left_beliefs = []
        right_beliefs = []
        image_feats = []

        for i in range(sequence_len):
            images_i = images[:,i].to(self.device)
            image_i_feat = self.resnet(images_i)
            image_i_feat = image_i_feat.view(batch_size, 512)
            if poses is not None:
                poses_i = poses[:,i].float().to(self.device)
                poses_i_feat = self.relu(self.pose_ff(poses_i))
                image_i_feat = torch.cat([image_i_feat, poses_i_feat], 1)
            if gazes is not None:
                gazes_i = gazes[:,i].float().to(self.device)
                gazes_i_feat = self.relu(self.gaze_ff(gazes_i))
                image_i_feat = torch.cat([image_i_feat, gazes_i_feat], 1)
            if bboxes is not None:
                bboxes_i = bboxes[:,i].float().to(self.device)
                bboxes_i_feat = self.relu(self.bbox_ff(bboxes_i))
                image_i_feat = torch.cat([image_i_feat, bboxes_i_feat], 1)
            if ocr_tensor is not None:
                ocr_tensor = ocr_tensor.to(self.device)
                ocr_tensor_feat = self.relu(self.ocr_ff(ocr_tensor))
                image_i_feat = torch.cat([image_i_feat, ocr_tensor_feat], 1)
            image_feats.append(image_i_feat)

        image_feats = torch.permute(torch.stack(image_feats), (1,0,2))
        left_beliefs = self.left(self.dropout(image_feats))
        right_beliefs = self.right(self.dropout(image_feats))

        return left_beliefs, right_beliefs


class ResNetGRU(nn.Module):
    def __init__(self, device):
        super(ResNetGRU, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))
        input_dim = 512 # rgb
        # input_dim = 512 + 150 # pose
        # input_dim = 512 + 6 # gaze
        # input_dim = 512 + 108 # bbox
        # input_dim = 512 + 64 # ocr
        # input_dim = 512 + 108 + 64 # bbox + ocr
        # input_dim = 512 + 150 + 6 # pose + gaze
        # input_dim = 512 + 150 + 6 + 108 + 64 # all
        self.gru = nn.GRU(input_dim, 512, batch_first=True)
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
        # FFs
        self.left = nn.Linear(512, 27)
        self.right = nn.Linear(512, 27)
        # modality FFs
        self.pose_ff = nn.Linear(150, 150)
        self.gaze_ff = nn.Linear(6, 6)
        self.bbox_ff = nn.Linear(108, 108)
        self.ocr_ff = nn.Linear(729, 64)
        # others
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.device = device

    def forward(self, images, poses, gazes, bboxes, ocr_tensor):
        batch_size, sequence_len, channels, height, width = images.shape
        left_beliefs = []
        right_beliefs = []
        rnn_inp = []

        for i in range(sequence_len):
            images_i = images[:,i].to(self.device)
            rnn_i_feat = self.resnet(images_i)
            rnn_i_feat = rnn_i_feat.view(batch_size, 512)
            if poses is not None:
                poses_i = poses[:,i].float().to(self.device)
                poses_i_feat = self.relu(self.pose_ff(poses_i))
                rnn_i_feat = torch.cat([rnn_i_feat, poses_i_feat], 1)
            if gazes is not None:
                gazes_i = gazes[:,i].float().to(self.device)
                gazes_i_feat = self.relu(self.gaze_ff(gazes_i))
                rnn_i_feat = torch.cat([rnn_i_feat, gazes_i_feat], 1)
            if bboxes is not None:
                bboxes_i = bboxes[:,i].float().to(self.device)
                bboxes_i_feat = self.relu(self.bbox_ff(bboxes_i))
                rnn_i_feat = torch.cat([rnn_i_feat, bboxes_i_feat], 1)
            if ocr_tensor is not None:
                ocr_tensor = ocr_tensor.to(self.device)
                ocr_tensor_feat = self.relu(self.ocr_ff(ocr_tensor))
                rnn_i_feat = torch.cat([rnn_i_feat, ocr_tensor_feat], 1)
            rnn_inp.append(rnn_i_feat)

        rnn_inp = torch.permute(torch.stack(rnn_inp), (1,0,2))
        rnn_out, _ = self.gru(rnn_inp)
        left_beliefs = self.left(self.dropout(rnn_out))
        right_beliefs = self.right(self.dropout(rnn_out))

        return left_beliefs, right_beliefs


class ResNetConv1D(nn.Module):
    def __init__(self, device):
        super(ResNetConv1D, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))
        input_dim = 512 # rgb
        # input_dim = 512 + 150 # pose
        # input_dim = 512 + 6 # gaze
        # input_dim = 512 + 108 # bbox
        # input_dim = 512 + 64 # ocr
        # input_dim = 512 + 108 + 64 # bbox + ocr
        # input_dim = 512 + 150 + 6 # pose + gaze
        # input_dim = 512 + 150 + 6 + 108 + 64 # all
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=5, padding=4)
        # FFs
        self.left = nn.Linear(512, 27)
        self.right = nn.Linear(512, 27)
        # modality FFs
        self.pose_ff = nn.Linear(150, 150)
        self.gaze_ff = nn.Linear(6, 6)
        self.bbox_ff = nn.Linear(108, 108)
        self.ocr_ff = nn.Linear(729, 64)
        # others
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.device = device

    def forward(self, images, poses, gazes, bboxes, ocr_tensor):
        batch_size, sequence_len, channels, height, width = images.shape
        left_beliefs = []
        right_beliefs = []
        conv1d_inp = []

        for i in range(sequence_len):
            images_i = images[:,i].to(self.device)
            images_i_feat = self.resnet(images_i)
            images_i_feat = images_i_feat.view(batch_size, 512)
            if poses is not None:
                poses_i = poses[:,i].float().to(self.device)
                poses_i_feat = self.relu(self.pose_ff(poses_i))
                images_i_feat = torch.cat([images_i_feat, poses_i_feat], 1)
            if gazes is not None:
                gazes_i = gazes[:,i].float().to(self.device)
                gazes_i_feat = self.relu(self.gaze_ff(gazes_i))
                images_i_feat = torch.cat([images_i_feat, gazes_i_feat], 1)
            if bboxes is not None:
                bboxes_i = bboxes[:,i].float().to(self.device)
                bboxes_i_feat = self.relu(self.bbox_ff(bboxes_i))
                images_i_feat = torch.cat([images_i_feat, bboxes_i_feat], 1)
            if ocr_tensor is not None:
                ocr_tensor = ocr_tensor.to(self.device)
                ocr_tensor_feat = self.relu(self.ocr_ff(ocr_tensor))
                images_i_feat = torch.cat([images_i_feat, ocr_tensor_feat], 1)
            conv1d_inp.append(images_i_feat)

        conv1d_inp = torch.permute(torch.stack(conv1d_inp), (1,2,0))
        conv1d_out = self.conv1d(conv1d_inp)
        conv1d_out = conv1d_out[:,:,:-4]
        conv1d_out = self.relu(torch.permute(conv1d_out, (0,2,1)))
        left_beliefs = self.left(self.dropout(conv1d_out))
        right_beliefs = self.right(self.dropout(conv1d_out))

        return left_beliefs, right_beliefs


class ResNetLSTM(nn.Module):
    def __init__(self, device):
        super(ResNetLSTM, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))
        input_dim = 512
        # input_dim = 512 + 150 # pose
        # input_dim = 512 + 6 # gaze
        # input_dim = 512 + 108 # bbox
        # input_dim = 512 + 64 # ocr
        # input_dim = 512 + 108 + 64 # bbox + ocr
        # input_dim = 512 + 150 + 6 # pose + gaze
        # input_dim = 512 + 150 + 6 + 108 + 64 # all
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=512, batch_first=True)
        # FFs
        self.left = nn.Linear(512, 27)
        self.right = nn.Linear(512, 27)
        # modality FFs
        self.pose_ff = nn.Linear(150, 150)
        self.gaze_ff = nn.Linear(6, 6)
        self.bbox_ff = nn.Linear(108, 108)
        self.ocr_ff = nn.Linear(729, 64)
        # others
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.device = device

    def forward(self, images, poses, gazes, bboxes, ocr_tensor):
        batch_size, sequence_len, channels, height, width = images.shape
        left_beliefs = []
        right_beliefs = []
        rnn_inp = []

        for i in range(sequence_len):
            images_i = images[:,i].to(self.device)
            rnn_i_feat = self.resnet(images_i)
            rnn_i_feat = rnn_i_feat.view(batch_size, 512)
            if poses is not None:
                poses_i = poses[:,i].float().to(self.device)
                poses_i_feat = self.relu(self.pose_ff(poses_i))
                rnn_i_feat = torch.cat([rnn_i_feat, poses_i_feat], 1)
            if gazes is not None:
                gazes_i = gazes[:,i].float().to(self.device)
                gazes_i_feat = self.relu(self.gaze_ff(gazes_i))
                rnn_i_feat = torch.cat([rnn_i_feat, gazes_i_feat], 1)
            if bboxes is not None:
                bboxes_i = bboxes[:,i].float().to(self.device)
                bboxes_i_feat = self.relu(self.bbox_ff(bboxes_i))
                rnn_i_feat = torch.cat([rnn_i_feat, bboxes_i_feat], 1)
            if ocr_tensor is not None:
                ocr_tensor = ocr_tensor.to(self.device)
                ocr_tensor_feat = self.relu(self.ocr_ff(ocr_tensor))
                rnn_i_feat = torch.cat([rnn_i_feat, ocr_tensor_feat], 1)
            rnn_inp.append(rnn_i_feat)

        rnn_inp = torch.permute(torch.stack(rnn_inp), (1,0,2))
        rnn_out, _ = self.lstm(rnn_inp)
        left_beliefs = self.left(self.dropout(rnn_out))
        right_beliefs = self.right(self.dropout(rnn_out))

        return left_beliefs, right_beliefs