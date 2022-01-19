# Here Im going to implement the model architecture of YOLOv1 and then the training, validation and testing process in PyTorch
# The model architecture is based on the paper: https://arxiv.org/abs/1506.02640
# Author : Morteza Heidari
# Date: January, 2022
# **********************************************************************************************************************
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import cv2
from utils import intersection_over_union, non_max_suppression,  mean_average_precision, cellboxes_to_boxes, get_bboxes, plot_image, save_checkpoint, load_checkpoint
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.utils.data import DataLoader
########################################################################################################################
# Global Variables
seed = 123
torch.manual_seed(seed)
LEARNIN_RATE = 2e-5
DEVISE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0 # we set it to zero since it is hard to actually to train the whole model, we can use the pretrained on IMAGE_NET, -> computational reasons
EPOCHS = 200
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
TRAIN_IMG_DIR = "archive/images"
TRAIN_LABEL_DIR = "archive/labels"
TRAIN_CSV = "archive/train.csv"
########################################################################################################################       


# **************************
# architecture configurations:
architecture_configurations = [
    (7, 64, 2, 3),  # Tuple: curnal size = 7, number of output filters = 64,  stride = 2, padding size = 3
    "M",  # M = maxpooling
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],  # 4 = number of repeat
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]
# **************************
# model class


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)  # bias=False is important since we are going to use batch norm
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class YOLOv1(nn.Module):
    # def __init__(self, in_channels, out_channels, kernel_size, stride, padding): # in_channels = 3 is the default value for RGB images
    def __init__(self, in_channels=3, **kwargs):  # in_channels = 3 is the default value for RGB images
        super(YOLOv1, self).__init__()
        self.architecture = architecture_configurations
        self.in_channels = in_channels
        self.darknet = self._create_darknet(self.architecture)
        # self.fully_connected = self._create_fully_connected(out_channels, kernel_size, stride, padding)
        self.fully_connected = self._create_fully_connected(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        # x = torch.flatten(x, start_dim=1)
        # x = self.fully_connected(x)
        return self.fully_connected(torch.flatten(x, start_dim=1))

    # now we are going to create the darknet architecture _create_darknet function
    def _create_darknet(self, architecture):
        layers = []
        in_channels = self.in_channels
        for layer in architecture:
            if type(layer) == tuple:
                layers.append(CNNBlock(in_channels, layer[1], kernel_size=layer[0], stride=layer[2], padding=layer[3],))
                in_channels = layer[1]
            elif layer == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif type(layer) == list:
                cnv1 = layer[0]  # which is a tuple
                cnv2 = layer[1]  # which is a tuple
                repeats = layer[2]
                for i in range(repeats):
                    layers.append(CNNBlock(in_channels, cnv1[1], kernel_size=cnv1[0], stride=cnv1[2], padding=cnv1[3],))
                    layers.append(CNNBlock(cnv1[1], cnv2[1], kernel_size=cnv2[0], stride=cnv2[2], padding=cnv2[3],))
                    in_channels = cnv2[1]
        return nn.Sequential(*layers)
    # In this part we are going to create the fully connected layer _create_fully_connected function

    def _create_fully_connected(self, split_size, n_boxes, n_classes):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * split_size * split_size, 4096),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(4096, split_size * split_size * (n_classes + n_boxes * 5)),  # we have to reshape it to (split_size, split_size, n_classes + n_boxes * 5 = 30)
        )

# ######################################################################################################################
# ************************** LOSS FUNCTION *****************************************************************************
# ######################################################################################################################
# loss class
# loss function in YOLOv1 is a combination of the following loss functions:
# boxes coordinates are normalized to the range [0, 1] and the midpoints are used as a part of loss function
# to be optimized to the best coordinates. Width and Haights are also used in loss function to be optimized to the.
# if there is or there is not object in the box, loss function is used to penalize the network for producing small.
# ######################################################################################################################


class Mloss(nn.Module): # loss function in page 4 of the original paper : https://arxiv.org/pdf/1506.02640.pdf
    def __init__(self, split_size=7, n_boxes=2, n_classes=20):
        super(Mloss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")  # reduction = "sum" is important since we are going to use it in the loss function
        self.S = split_size
        self.B = n_boxes
        self.C = n_classes
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.lambda_obj = 1

    def forward(self, pred, target):
        pred = pred.reshape(-1, self.S, self.S, self.C + self.B * 5)
        IOU_bb1 = intersection_over_union(pred[..., 21:25], target[..., 21:25])  # 21:25 is the coordinates of the boxes
        IOU_bb2 = intersection_over_union(pred[..., 26:30], target[..., 21:25])  # 25:29 is the coordinates of the boxes
        IOUs = torch.cat([IOU_bb1.unsqueeze(0), IOU_bb2.unsqueeze(0)], dim=0)
        IOUs_max, bestbox = torch.max(IOUs, dim=0)
        IOUs_bestbox2 = torch.where(IOUs_max == IOUs, IOUs_max, torch.zeros_like(IOUs_max))
        exist_box = target[..., 20].unsqueeze(3)  # Iobs_i is the existence of the box
        #  Now based on box coordinates and existence of the box we are going to calculate the loss
        box_predictions = exist_box * (bestbox * pred[..., 26:30] + (1 - bestbox) * pred[..., 21:25])
        box_targets = exist_box * target[..., 21:25]
        box_predictions[..., 2:4] = torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-9)) * torch.sign(box_predictions[..., 2:4])
        box_targets[..., 2:4] = torch.sqrt(torch.abs(box_targets[..., 2:4] + 1e-9))
        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2), torch.flatten(box_targets, end_dim=-2))

        # Now we are going to calculate the object loss:
        pred_box = bestbox * pred[..., 25:26] + (1 - bestbox) * pred[..., 20:21]
        object_loss = self.mse(torch.flatten(exist_box * pred_box), torch.flatten(exist_box * target[..., 20:21]))
        # Now we are going to calculate the no object loss:
        # (N, S, S, 1) -> (N, S*S)
        no_object_loss = self.mse(torch.flatten((1 - exist_box) * pred[..., 20:21], start_dim=1), torch.flatten((1 - exist_box) * target[..., 20:21], start_dim=1))
        no_object_loss += self.mse(torch.flatten((1 - exist_box) * pred[..., 25:26], start_dim=1), torch.flatten((1 - exist_box) * target[..., 20:21], start_dim=1))
        # Now we are going to calculate the class loss:
        # (N, S, S, n_classes) -> (N*S*S, n_classes)
        class_loss = self.mse(torch.flatten(exist_box * pred[..., :20], end_dim=-2), torch.flatten(exist_box * target[..., :20], end_dim=-2))
        # now we are going to calculate actual loss out of the above loss functions:
        loss = self.lambda_coord * box_loss + self.lambda_noobj * no_object_loss + self.lambda_obj * object_loss + class_loss
        return loss


# # **************************
# here we define class of dealing with PascalVOC dataset, to do so we inherit from torch.utils.data.Dataset
# the dataset is avaliable here : https://www.kaggle.com/dataset/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2
class PascalVOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S = 7, B=2, C=20, transform = None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform
        self.data = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label_path = os.path.join(self.label_dir, self.data.iloc[idx, 1])
        boxes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                classlabel, x, y, w, h = [float(x) if float(x) != int(float(x)) else int(x) for x in line.replace("\n", "").split()]
                boxes.append([classlabel, x, y, w, h])
        image_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(image_path)
        boxes = torch.tensor(boxes) 

        if self.transform:
            image, boxes = self.transform(image, boxes)
        label_matrix = torch.zeros(self.S, self.S, self.C + self.B * 5)
        for box in boxes:
            classlabel, x, y, w, h = box.tolist()
            classlabel = int(classlabel)
            # find out which cell row and column the box belongs to
            row = int(y * self.S)
            col = int(x * self.S)
            x_cell = self.S * x - col
            y_cell = self.S * y - row
            w_cell = self.S * w # scale the width of image appropriately with the number of cells in each row and column to achieve width of cell
            h_cell = self.S * h
            if label_matrix[row, col, 20] ==0:
                label_matrix[row, col, 20] = 1
                box_coords = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                label_matrix[row, col, 21:25] = box_coords
                label_matrix[row, col, classlabel] = 1
        return image, label_matrix
#*************************
# # trainings class
# we are going to specify how training should be done and all of the steps

# class

# **************************
# Here we are going to define a class for data augmentation which is more general than the typical method and applicable to the label side as well.
# class compose is the one helpful to this purpose.

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes # we apply transforms only on img # we are going to just resize the image to 448x448
        return img, bboxes

transforms = Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # we normalize the image to be between 0 and 1
])

# now by defining a training function we can call training loader, the model, optimizor, and the loss function to train the model
def train_function(model, optimizer, loss_function, train_loader):
    loop = tqdm(train_loader, leave = False)
    mean_loss = []
    for i, (img, label) in enumerate(loop):
        img = img.to(DEVISE)
        label = label.to(DEVISE)
        optimizer.zero_grad() # we are going to zero the gradients
        pred = model(img) 
        loss = loss_function(pred, label) # we are going to calculate the loss
        loss.backward() # we are going to backpropagate the loss
        optimizer.step() # we are going to update the weights
        mean_loss.append(loss.item())
        # loop.set_description(f"Loss: {np.mean(mean_loss):.4f}") # we are going to set the description of the progress bar
        loop.set_postfix(loss = loss.item()) # update the progress bar
    
    print(f" Here is what we get for loss :  {np.mean(mean_loss):.4f}")

def test(Split_size=7, N_boxes=2, N_classes=20):
    model = YOLOv1(split_size=Split_size, n_boxes=N_boxes, n_classes=N_classes)
    x = torch.randn((5, 3, 448, 448))
    print(model(x).shape)


def main():
    # First step: initialize the model 
    model = YOLOv1(split_size=7, n_boxes=2, n_classes=20).to(DEVISE) 
    # SECOND STEP: initialize the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNIN_RATE, weight_decay=WEIGHT_DECAY)
    loss_function = Mloss()
    # THIRD STEP: check if the load_model is True or False
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer) # if we have a model saved
    # FOURTH STEP: we are going to creat a dataset for training

    # train_dataset = PascalVOCDataset(csv_file = TRAIN_CSV, img_dir = TRAIN_IMG_DIR, label_dir = TRAIN_LABEL_DIR, S = 7, B = 2, C = 20, transform = transforms)
    train_dataset = PascalVOCDataset(csv_file = "archive/100examples.csv", img_dir = TRAIN_IMG_DIR, label_dir = TRAIN_LABEL_DIR, S = 7, B = 2, C = 20, transform = transforms)
    # FIFTH STEP: we are going to creat a dataset for validation

    # val_dataset = PascalVOCDataset(csv_file = TRAIN_CSV, img_dir = TRAIN_IMG_DIR, label_dir = TRAIN_LABEL_DIR, S = 7, B = 2, C = 20, transform = transforms) #
    val_dataset = PascalVOCDataset(csv_file = "archive/test.csv", img_dir = TRAIN_IMG_DIR, label_dir = TRAIN_LABEL_DIR, S = 7, B = 2, C = 20, transform = transforms) #

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS, pin_memory= PIN_MEMORY, drop_last = False) # drop last is true because we are going to check if the last batch does not have enough data just drop it
    test_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS, pin_memory= PIN_MEMORY, drop_last = False) # drop last is true because we are going to check if the last batch does not have enough data just drop it
    # SIXTH STEP: we are going to train the model
    for epoch in range(EPOCHS):
        # for x, y in train_loader:
        #    x = x.to(DEVISE)
        #    for idx in range(8):
        #        bboxes = cellboxes_to_boxes(model(x))
        #        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #        plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

        #    import sys
        #    sys.exit()
        #****************************
        prediction_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold= 0.5, threshold=0.4)
        mean_avg_pre = mean_average_precision(prediction_boxes, target_boxes, iou_threshold=  0.5, box_format = "midpoint")
        print(f"Mean average precision: {mean_avg_pre:.4f}")
        # if mean_avg_pre > 0.95:
        #     checkpoint = {
        #         "model_state_dict": model.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict()
        #     }
        #     torch.save(checkpoint, "model_checkpoints/model_checkpoint.pt")
        train_function(model, optimizer, loss_function, train_loader)


if __name__ == "__main__":
    main()
