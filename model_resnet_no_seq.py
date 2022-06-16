import torch
import torch.nn as nn
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        # Conv
        resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-2]))
        # FFs
        # inp_dim = 9152 + 150 # pose
        # inp_dim = 9152 + 108 # bbox
        inp_dim = 9152 + 64 # ocr
        self.ff1 = nn.Linear(inp_dim, 512)
        self.ff2 = nn.Linear(512, 256)
        self.left = nn.Linear(256, 27)
        self.right = nn.Linear(256, 27)
        # modality FFs
        self.pose_ff = nn.Linear(150, 150)
        self.bbox_ff = nn.Linear(108, 108)
        self.ocr_ff = nn.Linear(729, 64)
        # others
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.device = device

    def forward(self, images, poses, gazes, bboxes, ocr_tensor):
        batch_size, channels, height, width = images.shape
        left_beliefs = []
        right_beliefs = []
        image_feats = []

        images_i = images.to(self.device)
        image_i_feat = self.resnet(images_i)
        if poses is not None:
            poses_i = poses.float().to(self.device)
            poses_i_feat = self.relu(self.pose_ff(poses_i))
            poses_i_feat = poses_i_feat.view(batch_size, 150, 1, 1).expand(-1, -1, image_i_feat.shape[2], image_i_feat.shape[3])
            image_i_feat = torch.cat([image_i_feat, poses_i_feat], 1)
        if bboxes is not None:
            bboxes_i = bboxes.float().to(self.device)
            bboxes_i_feat = self.relu(self.bbox_ff(bboxes_i))
            bboxes_i_feat = bboxes_i_feat.view(batch_size, 108, 1, 1).expand(-1, -1, image_i_feat.shape[2], image_i_feat.shape[3])
            image_i_feat = torch.cat([image_i_feat, bboxes_i_feat], 1)
        if ocr_tensor is not None:
            ocr_tensor = ocr_tensor.to(self.device)
            ocr_tensor_feat = self.relu(self.ocr_ff(ocr_tensor))
            ocr_tensor_feat = ocr_tensor_feat.view(batch_size, 64, 1, 1).expand(batch_size, -1, image_i_feat.shape[2], image_i_feat.shape[3])
            image_i_feat = torch.cat([image_i_feat, ocr_tensor_feat], 1)
        
        # image_feats.append(image_i_feat)
        # image_feats = torch.stack(image_feats)
        # print(image_feats.shape, image_i_feat.shape)
        # image_feats = torch.flatten(image_feats, 1)
        image_feats = torch.flatten(image_i_feat, 1)
        x = self.relu(self.ff1(image_feats))
        x = self.dropout(x)
        x = self.relu(self.ff2(x))
        x = self.dropout(x)
        left_beliefs = self.left(x)
        right_beliefs = self.right(x)

        return left_beliefs, right_beliefs