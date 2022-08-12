import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, input_dim, device):
        super(ResNet, self).__init__()
        # Conv
        resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))
        # FFs
        self.left = nn.Linear(input_dim, 27)
        self.right = nn.Linear(input_dim, 27)
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
    def __init__(self, input_dim, device):
        super(ResNetGRU, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))
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
    def __init__(self, input_dim, device):
        super(ResNetConv1D, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))
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
    def __init__(self, input_dim, device):
        super(ResNetLSTM, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))
        # self.lstm = nn.LSTM(input_size=input_dim, hidden_size=512, batch_first=True)
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

    def forward(self, images, poses, gazes, bboxes, ocr_tensor, labels):
        batch_size, sequence_len, channels, height, width = images.shape
        left_beliefs = []
        right_beliefs = []
        rnn_inp = []
        
        mm_loss = 0
        for i in range(sequence_len):
            images_i = images[:,i].to(self.device)
            rnn_i_feat = self.resnet(images_i)
            rnn_i_feat = [rnn_i_feat.view(batch_size, 512)]
            modal_num = 1
            if i == 0:
                in_dim = [512]
                if poses is not None:
                    in_dim.append(150)
                    modal_num += 1
                if gazes is not None:
                    in_dim.append(6)
                    modal_num += 1
                if bboxes is not None:
                    in_dim.append(108)
                    modal_num += 1
                if ocr_tensor is not None:
                    in_dim.append(729)
                    ocr_tensor = ocr_tensor.to(self.device)
                    modal_num += 1
                self.mmdynamic = MMDynamicFuse(in_dim, [64], 27, dropout=0.5)
                self.lstm = nn.LSTM(input_size=64*modal_num, hidden_size=512, batch_first=True)

            if poses is not None:
                rnn_i_feat.append(poses[:,i].float().to(self.device))
            if gazes is not None:
                rnn_i_feat.append(gazes[:,i].float().to(self.device))
            if bboxes is not None:
                rnn_i_feat.append(bboxes[:,i].float().to(self.device))
            if ocr_tensor is not None:
                rnn_i_feat.append(ocr_tensor)
            rnn_i_feat, mm_loss_i = self.mmdynamic(rnn_i_feat, labels[:,i,:])
            mm_loss += mm_loss_i

            # if poses is not None:
            #     poses_i = poses[:,i].float().to(self.device)
            #     poses_i_feat = self.relu(self.pose_ff(poses_i))
            #     rnn_i_feat = torch.cat([rnn_i_feat, poses_i_feat], 1)
            # if gazes is not None:
            #     gazes_i = gazes[:,i].float().to(self.device)
            #     gazes_i_feat = self.relu(self.gaze_ff(gazes_i))
            #     rnn_i_feat = torch.cat([rnn_i_feat, gazes_i_feat], 1)
            # if bboxes is not None:
            #     bboxes_i = bboxes[:,i].float().to(self.device)
            #     bboxes_i_feat = self.relu(self.bbox_ff(bboxes_i))
            #     rnn_i_feat = torch.cat([rnn_i_feat, bboxes_i_feat], 1)
            # if ocr_tensor is not None:
            #     ocr_tensor = ocr_tensor.to(self.device)
            #     ocr_tensor_feat = self.relu(self.ocr_ff(ocr_tensor))
            #     rnn_i_feat = torch.cat([rnn_i_feat, ocr_tensor_feat], 1)

            rnn_inp.append(rnn_i_feat)

        print(rnn_inp.shape)
        rnn_inp = torch.permute(torch.stack(rnn_inp), (1,0,2))
        rnn_out, _ = self.lstm(rnn_inp)
        left_beliefs = self.left(self.dropout(rnn_out))
        right_beliefs = self.right(self.dropout(rnn_out))

        return left_beliefs, right_beliefs, mm_loss





def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class MMDynamicFuse(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.FeatureInforEncoder = nn.ModuleList([LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayerLeft = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPConfidenceLayerRight = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayerLeft = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.TCPClassifierLayerRight = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        # self.MMClasifier = []
        # for layer in range(1, len(hidden_dim)-1):
        #     self.MMClasifier.append(LinearLayer(self.views*hidden_dim[0], hidden_dim[layer]))
        #     self.MMClasifier.append(nn.ReLU())
        #     self.MMClasifier.append(nn.Dropout(p=dropout))
        # if len(self.MMClasifier):
        #     self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        # else:
        #     self.MMClasifier.append(LinearLayer(self.views*hidden_dim[-1], num_class))
        # self.MMClasifier = nn.Sequential(*self.MMClasifier)


    def forward(self, data_list, label=None):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, featureLeft, featureRight, TCPLogitLeft, TCPLogitRight, TCPConfidenceLeft, TCPConfidenceRight = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()
        for view in range(self.views):
            FeatureInfo[view] = torch.sigmoid(self.FeatureInforEncoder[view](data_list[view]))
            feature[view] = data_list[view] * FeatureInfo[view]
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            TCPLogitLeft[view] = self.TCPClassifierLayerLeft[view](feature[view])
            TCPLogitRight[view] = self.TCPClassifierLayerRight[view](feature[view])
            TCPConfidenceLeft[view] = self.TCPConfidenceLayerLeft[view](feature[view])
            TCPConfidenceRight[view] = self.TCPConfidenceLayerRight[view](feature[view])
            featureLeft[view] = feature[view] * TCPConfidenceLeft[view]
            featureRight[view] = feature[view] * TCPConfidenceRight[view]

        for view in range(self.views):
            # MMLoss = MMLoss+torch.mean(FeatureInfo[view])
            MMLoss = torch.mean(FeatureInfo[view])
            # left
            predLeft = F.softmax(TCPLogitLeft[view], dim=1)
            # print(predLeft, predLeft.shape, label.shape)
            left_label = label[:,0]
            p_target = torch.gather(input=predLeft, dim=1, index=left_label.unsqueeze(dim=1)).view(-1)
            # print(p_target.shape, TCPConfidenceLeft[view].shape, TCPConfidenceLeft[view].view(-1).shape, TCPLogitLeft[view].shape, left_label.shape)
            left_confidence_loss = torch.mean(F.mse_loss(TCPConfidenceLeft[view].view(-1), p_target)+criterion(TCPLogitLeft[view], left_label))
            # right
            predRight = F.softmax(TCPLogitRight[view], dim=1)
            right_label = label[:,1]
            p_target = torch.gather(input=predRight, dim=1, index=right_label.unsqueeze(dim=1)).view(-1)
            right_confidence_loss = torch.mean(F.mse_loss(TCPConfidenceRight[view].view(-1), p_target)+criterion(TCPLogitRight[view], right_label))
            MMLoss = MMLoss+left_confidence_loss+right_confidence_loss
        # return MMLoss, MMlogit

        MMfeature = torch.cat([i for i in feature.values()], dim=1)

        return MMfeature, MMLoss

# class MMDynamic(nn.Module):
#     def __init__(self, in_dim, hidden_dim, num_class, dropout):
#         super().__init__()
#         self.views = len(in_dim)
#         self.classes = num_class
#         self.dropout = dropout

#         self.FeatureInforEncoder = nn.ModuleList([LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
#         self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
#         self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
#         self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

#         self.MMClasifier = []
#         for layer in range(1, len(hidden_dim)-1):
#             self.MMClasifier.append(LinearLayer(self.views*hidden_dim[0], hidden_dim[layer]))
#             self.MMClasifier.append(nn.ReLU())
#             self.MMClasifier.append(nn.Dropout(p=dropout))
#         if len(self.MMClasifier):
#             self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
#         else:
#             self.MMClasifier.append(LinearLayer(self.views*hidden_dim[-1], num_class))
#         self.MMClasifier = nn.Sequential(*self.MMClasifier)


#     def forward(self, data_list, label=None, infer=False):
#         criterion = torch.nn.CrossEntropyLoss(reduction='none')
#         FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()
#         for view in range(self.views):
#             FeatureInfo[view] = torch.sigmoid(self.FeatureInforEncoder[view](data_list[view]))
#             feature[view] = data_list[view] * FeatureInfo[view]
#             feature[view] = self.FeatureEncoder[view](feature[view])
#             feature[view] = F.relu(feature[view])
#             feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
#             TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
#             TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])
#             feature[view] = feature[view] * TCPConfidence[view]

#         MMfeature = torch.cat([i for i in feature.values()], dim=1)
#         MMlogit = self.MMClasifier(MMfeature)
#         if infer:
#             return MMlogit
#         MMLoss = torch.mean(criterion(MMlogit, label))
#         for view in range(self.views):
#             MMLoss = MMLoss+torch.mean(FeatureInfo[view])
#             pred = F.softmax(TCPLogit[view], dim=1)
#             p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
#             confidence_loss = torch.mean(F.mse_loss(TCPConfidence[view].view(-1), p_target)+criterion(TCPLogit[view], label))
#             MMLoss = MMLoss+confidence_loss
#         return MMLoss, MMlogit
    
#     def infer(self, data_list):
#         MMlogit = self.forward(data_list, infer=True)
#         return MMlogit