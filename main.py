import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
import torchvision
from torchvision.transforms import functional as F
from pathlib import Path


class RegionProposalNetwork(nn.Module):

    def __init__(self, in_channels, mid_channels, anchor_ratios = [0.5,1,2], anchor_scales = [8,16,32]):

        super(RegionProposalNetwork, self).__init__()

        self.n_anchors = len(anchor_ratios) * len(anchor_scales) #number of anchors based on anchor variations

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding = 1) #feature map
 
        self.score_conv = nn.Conv2d(mid_channels, self.n_anchors*2,  kernel_size = 1) #objectness score

        self.bbox_conv = nn.Conv2d(mid_channels, self.n_anchors * 4, kernel_size = 1) #bounding box adjustments

        self._initialize_weights()

    
    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight) #xavier initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    
    def forward(self, x):

        h = F.relu(self.conv1(x))

        rpn_scores = self.score_conv(h)  #objectness score 
        rpn_scores = rpn_scores.permute(0,2,3,1).contiguous()
        rpn_scores = rpn_scores.view(x.size(0), -1,2)


        rpn_bbox = self.bbox_conv(h) #bounding box adjustment
        rpn_bbox = rpn_bbox.permute(0,2,3,1).contiguous()
        rpn_bbox = rpn_bbox.view(x.size(0), -1, 4)

        return rpn_scores, rpn_bbox
    

class FasterRCNN(nn.module):

    def __init__(self, num_classes, backbone = 'resnet50'):

        super(FasterRCNN, self).__init__()

        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained = True)
            backbone_out_channels = 2048

        else:
             raise ValueError(f'unsupported backbone')
        
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.rpn = RegionProposalNetwork(
            in_channels = backbone_out_channels,
            mid_channels = 512
        )

        self.roi_pool = nn.AdaptiveMaxPool2d((7,7))

        self.detection_head = nn.Sequential(
            nn.Linear(backbone_out_channels * 7 *7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)

        )

        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)

    
    def forward(self, x):
        
        features = self.backbone(x)

        rpn_scores, rpn_bbox = self.rpn(features)

        batch_size = x.size(0)
        dummy_rois = torch.zeros(batch_size, 100,4).to(x.device)


        pooled_features = []
        for i in range(batch_size):
            pooled = self.roi_pool(features[i:i+1])
            pooled_features.append(pooled)

        pooled_features = torch.cat(pooled_features, dim = 0)

        pooled_features = pooled_features.view(batch_size, -1)

        roi_features = self.detection_head(pooled_features)

        cls_scores = self.cls_score(roi_features)
        bbox_preds = self.bbox_pred(roi_features)

        return cls_scores, bbox_preds, rpn_scores, rpn_bbox
    


class customDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform = None):

        self.annotations = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform


        self.unique_frames = self.annotations['frame'].unique()
        self.frame_to_annotations = self.annotations.groupby('frame')

    def __len__(self):
        return len(self.unique_frames)
    
    def __getitem__(self, idx):
        frame_name = self.unique_frames[idx]

        img_path = self.img_dir / frame_name
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        frame_annotations = self.frame_to_annotations.get_group(frame_name)



        boxes = []
        labels = []
        for _, row in frame_annotations.iterrows():
            boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            labels.append(row['class_id'])

        
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        labels = torch.as_tensor(labels, dtype = torch.int64)


        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])

        }

        image = F.to_tensor(image)

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target
    

def collate_fn(batch):

    return tuple(zip(*batch))
    
def calculate_loss(cls_scores, bbox_preds, rpn_scores, rpn_bbox, targets):

    rpn_cls_loss = F.cross_entropy(rpn_scores, targets['rpn_labels'])

    rpn_loc_loss = F.smooth_l1_loss(rpn_bbox, targets['rpn_bbox_targets'])

    det_cls_loss = F.cross_entropy(cls_scores, targets['labels'])

    det_loc_loss = F.smooth_l1_loss(bbox_preds, targets['bbox_targets'])

    total_loss = rpn_cls_loss + rpn_loc_loss + det_cls_loss + det_loc_loss

    return total_loss
    
def train_one_epoch(model, data_loader, optimizer, device = 'cpu'):

    model.train()

    total_loss = 0

    for images, targets in data_loader:

        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] 

        optimizer.zero_grad()

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)
    

def main(img_dir, csv_path, num_epochs = 10, batch_size = 2):

    device = 'cpu'

    dataset = customDataset(
        csv_file = csv_path,
        img_dir = img_dir
                            )
            
    data_loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = collate_fn,
        num_workers = 4
    )

    num_classes = len(dataset.annotations['class_id'].unique()) + 1
    model = FasterRCNN(num_classes)
    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr = 0.005,
        momentum = 0.9,
        weight_decay = 0.0005
    )

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size = 3,
        gamma = 0.1
    )

    for epoch in range(num_epochs):
        
        print(f'epoch {epoch + 1} / {num_epochs}')

        loss = train_one_epoch(model ,data_loader, optimizer, device )

        print(f'loss: {loss:.4f}')

        lr_scheduler.step()

        if (epoch + 1) % 5 == 0:

            checkpoint = {

                'epoch': epoch + 1,
                'model_state_dict': optimizer.state_dict(),
                'loss': loss,

            }

            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')


        torch.save(model.state_dict(), 'faster_rcnn_final.pth')
        return model
    

if __name__ == "__main__":

    IMG_DIR = "C:\\Users\\77204\\Desktop\\Python\\computer-vision\\project1\\archive (9)\\images"

    csv_path = "C:\\Users\\77204\\Desktop\\Python\\computer-vision\\project1\\archive (9)\\labels_train.csv"

    train_model = main(

        img_dir = IMG_DIR,
        csv_path = csv_path,
        num_epochs = 10,
        batch_size = 2 
    
    )





