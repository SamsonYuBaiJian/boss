import torch
import torch.nn as nn
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        # Conv
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
        # self.ff1 = nn.Linear(512*4*4, 512)
        # self.ff2 = nn.Linear(512, 256)
        self.left = nn.Linear(512, 27)
        self.right = nn.Linear(512, 27)
        # modality FFs
        self.pose_ff = nn.Linear(150, 150)
        # self.gaze_ff = nn.Linear(6, 6)
        self.gaze_ff = nn.Linear(3, 64)
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
                # poses_i_feat = poses_i_feat.view(batch_size, 150, 1, 1).expand(batch_size, -1, images_i_feat.shape[2], images_i_feat.shape[3])
                images_i_feat = torch.cat([images_i_feat, poses_i_feat], 1)
            if gazes is not None:
                gazes_i = gazes[:,i].float().to(self.device)
                gazes_i_feat = self.relu(self.gaze_ff(gazes_i))
                # poses_i_feat = poses_i_feat.view(batch_size, 150, 1, 1).expand(-1, -1, image_i_feat.shape[2], image_i_feat.shape[3])
                images_i_feat = torch.cat([images_i_feat, gazes_i_feat], 1)
            if bboxes is not None:
                bboxes_i = bboxes[:,i].float().to(self.device)
                bboxes_i_feat = self.relu(self.bbox_ff(bboxes_i))
                # bboxes_i_feat = bboxes_i_feat.view(batch_size, 108, 1, 1).expand(batch_size, -1, images_i_feat.shape[2], images_i_feat.shape[3])
                images_i_feat = torch.cat([images_i_feat, bboxes_i_feat], 1)
            if ocr_tensor is not None:
                ocr_tensor = ocr_tensor.to(self.device)
                ocr_tensor_feat = self.relu(self.ocr_ff(ocr_tensor))
                # ocr_tensor_feat = ocr_tensor_feat.view(batch_size, 64, 1, 1).expand(batch_size, -1, images_i_feat.shape[2], images_i_feat.shape[3])
                images_i_feat = torch.cat([images_i_feat, ocr_tensor_feat], 1)
            conv1d_inp.append(images_i_feat)

        # conv3d_inp = torch.permute(torch.stack(conv3d_inp), (1,2,0,3,4))
        # print(torch.stack(conv1d_inp).shape)
        conv1d_inp = torch.permute(torch.stack(conv1d_inp), (1,2,0))
        conv1d_out = self.conv1d(conv1d_inp)
        # print(conv1d_out.shape, conv1d_inp.shape)
        conv1d_out = conv1d_out[:,:,:-4]
        conv1d_out = self.relu(torch.permute(conv1d_out, (0,2,1)))
        # conv3d_out = torch.permute(conv3d_out, (0,2,1,3,4))
        # conv3d_out = torch.flatten(conv3d_out, 2)
        # x = self.relu(self.ff1(conv3d_out))
        # x = self.dropout(x)
        # x = self.relu(self.ff2(x))
        # x = self.dropout(x)
        left_beliefs = self.left(self.dropout(conv1d_out))
        right_beliefs = self.right(self.dropout(conv1d_out))

        return left_beliefs, right_beliefs