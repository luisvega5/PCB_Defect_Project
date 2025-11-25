"""
    ======================================================================
                            CNN Model Script
    ======================================================================
    Name:          cnn_model.py
    Author:        Luis Obed Vega Maisonet
    University:    University of Central Florida (UCF)
    Course:        CAP 5415 - Computer Vision
    Academic Year: 2025
    Semester:      Fall
    Professor:     Dr. Yogesh Singh Rawat

    Description:
    This script defines the CNN architecture to classify PCB defects
    from images or patches. The main component is the DefectCNN
    class, which inherits from torch.nn.Module. Inside, the model is
    composed of several convolutional blocks, each block combining a
    convolution layer, batch normalization, ReLU activation, and max
    pooling to downsample features and increase the number of channels.
    After these convolutional layers, the script flattens the spatial
    feature maps into a single vector, passes them through a fully
    connected layer, applies a ReLU and dropout for regularization,
    and finally uses a last fully connected layer to produce logits
    for num_classes defect categories. The model assumes a fixed input
    resolution (for example 128×128), and its internal flatten_dim is
    set accordingly. This script encapsulates the learnable parameters
    and forward computation of a CNN classifier.
------------------------------------------------Imports and Globals-----------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
"""-----------------------------------------------Class Definition---------------------------------------------------"""
class DefectCNN(nn.Module):
    """
        Function Name: __init__
        Description:
            Initializes the DefectCNN model used for PCB defect classification.
            This method constructs three convolutional blocks (conv + batch
            normalization + ReLU + max pooling) followed by two fully connected
            layers with dropout regularization. It assumes 3-channel RGB inputs
            of fixed spatial resolution (e.g., 128×128) and configures the
            flatten_dim accordingly so that the final linear layer produces
            logits for the specified number of classes.
        Input:
            num_classes (int, optional):
                Number of defect categories the network should predict. This
                value controls the size of the final output layer. The default
                is 6 to match the six PCB defect types used in the project.
        Output:
            None.
    """
    """-------------------------------------------------Functions----------------------------------------------------"""
    def __init__(self, num_classes: int = 6):
        super().__init__()

        # Block 1: Convolution + ReLU + MaxPool
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # NOTE: The size of the flattened feature vector depends on the input image size.
        # For example, if input images are 128x128:
        #   After pool1: 32 x 64 x 64
        #   After pool2: 64 x 32 x 32
        #   After pool3: 128 x 16 x 16  -> 128 * 16 * 16 = 32768
        # Adjust this value if you change the input resolution.
        self.flatten_dim = 128 * 16 * 16

        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    """
        Function Name: forward
        Description:
            Defines the forward pass of the DefectCNN model. The input batch of
            images is processed sequentially through three convolutional blocks
            (each consisting of convolution, batch normalization, ReLU
            activation, and max pooling) to extract hierarchical features. The
            resulting feature maps are flattened and passed through a fully
            connected layer with ReLU and dropout, followed by a final linear
            layer that produces class logits for PCB defect prediction.
        Input:
            x (torch.Tensor):
                Batch of input images of shape (batch_size, 3, H, W), where
                each image is a 3-channel RGB image that has been preprocessed
                (e.g., resized and normalized) to match the expected input
                resolution of the network.
        Output:
            logits (torch.Tensor):
                Tensor of shape (batch_size, num_classes) containing the raw
                unnormalized scores for each defect class. These logits are
                typically passed to a softmax or cross-entropy loss function
                during training and can be converted to class probabilities
                during inference.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = self.pool3(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits