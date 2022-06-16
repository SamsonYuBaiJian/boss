import torch
import torch.nn as nn
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
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
        # self.ff1 = nn.Linear(inp_dim*4*4, 512)
        # self.ff2 = nn.Linear(512, 256)
        self.left = nn.Linear(inp_dim, 27)
        self.right = nn.Linear(inp_dim, 27)
        # modality FFs
        self.pose_ff = nn.Linear(150, 150)
        # self.gaze_ff = nn.Linear(6, 6)
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
                # poses_i_feat = poses_i_feat.view(batch_size, 150, 1, 1).expand(-1, -1, image_i_feat.shape[2], image_i_feat.shape[3])
                image_i_feat = torch.cat([image_i_feat, poses_i_feat], 1)
            if gazes is not None:
                gazes_i = gazes[:,i].float().to(self.device)
                gazes_i_feat = self.relu(self.gaze_ff(gazes_i))
                # poses_i_feat = poses_i_feat.view(batch_size, 150, 1, 1).expand(-1, -1, image_i_feat.shape[2], image_i_feat.shape[3])
                image_i_feat = torch.cat([image_i_feat, gazes_i_feat], 1)
            if bboxes is not None:
                bboxes_i = bboxes[:,i].float().to(self.device)
                bboxes_i_feat = self.relu(self.bbox_ff(bboxes_i))
                # bboxes_i_feat = bboxes_i_feat.view(batch_size, 108, 1, 1).expand(-1, -1, image_i_feat.shape[2], image_i_feat.shape[3])
                image_i_feat = torch.cat([image_i_feat, bboxes_i_feat], 1)
            if ocr_tensor is not None:
                ocr_tensor = ocr_tensor.to(self.device)
                ocr_tensor_feat = self.relu(self.ocr_ff(ocr_tensor))
                # ocr_tensor_feat = ocr_tensor_feat.view(batch_size, 64, 1, 1).expand(batch_size, -1, image_i_feat.shape[2], image_i_feat.shape[3])
                image_i_feat = torch.cat([image_i_feat, ocr_tensor_feat], 1)
            image_feats.append(image_i_feat)

        image_feats = torch.permute(torch.stack(image_feats), (1,0,2))
        # image_feats = torch.flatten(image_feats, 2)
        # print(image_feats.shape)
        left_beliefs = self.left(self.dropout(image_feats))
        right_beliefs = self.right(self.dropout(image_feats))

        return left_beliefs, right_beliefs