import torch
import torch.nn as nn
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, device, nc=3, nf=16):
        super(Model, self).__init__()
        # Conv
        resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))
        # RNNs
        gru_input_dim = 512
        # gru_input_dim = 512 + 150 # pose
        # gru_input_dim = 512 + 6 # gaze
        # gru_input_dim = 512 + 108 # bbox
        # gru_input_dim = 512 + 64 # ocr
        # gru_input_dim = 512 + 108 + 64 # bbox + ocr
        # gru_input_dim = 512 + 150 + 6 # pose + gaze
        # gru_input_dim = 512 + 150 + 6 + 108 + 64 # all
        self.gru = nn.GRU(gru_input_dim, 512, batch_first=True)
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
        # FFs
        # self.ff = nn.Linear(512, 256)
        # self.left = nn.Linear(256, 27)
        # self.right = nn.Linear(256, 27)
        self.left = nn.Linear(512, 27)
        self.right = nn.Linear(512, 27)
        # modality FFs
        self.pose_ff = nn.Linear(150, 150)
        # self.gaze_ff = nn.Linear(6, 6)
        # self.bbox_ff = nn.Linear(108, 108)
        # self.ocr_ff = nn.Linear(729, 64)
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
                # poses_i_feat = poses_i_feat.view(batch_size, 150, 1, 1).expand(-1, -1, rnn_i_feat.shape[2], rnn_i_feat.shape[3])
                rnn_i_feat = torch.cat([rnn_i_feat, poses_i_feat], 1)
            if gazes is not None:
                gazes_i = gazes[:,i].float().to(self.device)
                gazes_i_feat = self.relu(self.gaze_ff(gazes_i))
                # poses_i_feat = poses_i_feat.view(batch_size, 150, 1, 1).expand(-1, -1, image_i_feat.shape[2], image_i_feat.shape[3])
                rnn_i_feat = torch.cat([rnn_i_feat, gazes_i_feat], 1)
            if bboxes is not None:
                bboxes_i = bboxes[:,i].float().to(self.device)
                bboxes_i_feat = self.relu(self.bbox_ff(bboxes_i))
                # bboxes_i_feat = bboxes_i_feat.view(batch_size, 108, 1, 1).expand(-1, -1, rnn_i_feat.shape[2], rnn_i_feat.shape[3])
                rnn_i_feat = torch.cat([rnn_i_feat, bboxes_i_feat], 1)
            if ocr_tensor is not None:
                ocr_tensor = ocr_tensor.to(self.device)
                ocr_tensor_feat = self.relu(self.ocr_ff(ocr_tensor))
                # ocr_tensor_feat = ocr_tensor_feat.view(batch_size, 64, 1, 1).expand(batch_size, -1, rnn_i_feat.shape[2], rnn_i_feat.shape[3])
                rnn_i_feat = torch.cat([rnn_i_feat, ocr_tensor_feat], 1)
            rnn_inp.append(rnn_i_feat)

        # rnn_inp = torch.permute(torch.stack(rnn_inp), (1,0,2,3,4))
        rnn_inp = torch.permute(torch.stack(rnn_inp), (1,0,2))
        # rnn_inp = torch.flatten(rnn_inp, 2)
        rnn_out, _ = self.gru(rnn_inp)
        # x = self.ff(rnn_out)
        # x = self.relu(x)
        # x = self.dropout(x)
        left_beliefs = self.left(self.dropout(rnn_out))
        right_beliefs = self.right(self.dropout(rnn_out))

        return left_beliefs, right_beliefs