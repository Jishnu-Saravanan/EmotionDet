import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchsummary import summary

import os
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import cv2

import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Defining batch size
batch_size = 32

# Transformations for training and validation sets
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# Load datasets through first method
dataset = datasets.ImageFolder(
    "C:/Users/jishn/OneDrive/Desktop/Emotion Detection Project/Datasets/FER2013/train",
    transform=transform,
)
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# # Load datasets through second method
# class CustomImageDataset(Dataset):
#     def __init__(self, csv_file, img_dir, transform=None):
#         self.labels_frame = pd.read_csv(csv_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         # Define the columns to be used as labels
#         self.label_columns = [
#             "neutral",
#             "happiness",
#             "surprise",
#             "sadness",
#             "anger",
#             "disgust",
#             "fear",
#             "contempt",
#             "unknown",
#             "NF",
#         ]

#     def __len__(self):
#         return len(self.labels_frame)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         # Get the image name and label values
#         img_name = os.path.join(self.img_dir, self.labels_frame.iloc[idx, 0])
#         image = Image.open(img_name).convert("L")  # Convert image to grayscale

#         # Extract the relevant label columns and find the index of the max value
#         labels = self.labels_frame.loc[idx, self.label_columns].values
#         label = labels.argmax()

#         if self.transform:
#             image = self.transform(image)

#         return image, label


# # Paths to the CSV file and image directory
# train_csv_file = "C:/Users/jishn/OneDrive/Desktop/Emotion Detection Project/Datasets/FER2013/FER2013Train/label.csv"
# train_img_dir = "C:/Users/jishn/OneDrive/Desktop/Emotion Detection Project/Datasets/FER2013/FER2013Train"

# test_csv_file = "C:/Users/jishn/OneDrive/Desktop/Emotion Detection Project/Datasets/FER2013/FER2013Test/label.csv"
# test_img_dir = "C:/Users/jishn/OneDrive/Desktop/Emotion Detection Project/Datasets/FER2013/FER2013Test"

# val_csv_file = "C:/Users/jishn/OneDrive/Desktop/Emotion Detection Project/Datasets/FER2013/FER2013Valid/label.csv"
# val_img_dir = "C:/Users/jishn/OneDrive/Desktop/Emotion Detection Project/Datasets/FER2013/FER2013Valid"

# # Define transformations for grayscale images resized to 48x48
# transform = transforms.Compose(
#     [
#         transforms.Grayscale(num_output_channels=1),
#         transforms.Resize((48, 48)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,)),
#     ]
# )

# # Instantiate the dataset
# train_dataset = CustomImageDataset(
#     csv_file=train_csv_file, img_dir=train_img_dir, transform=transform
# )

# test_dataset = CustomImageDataset(
#     csv_file=test_csv_file, img_dir=test_img_dir, transform=transform
# )

# val_dataset = CustomImageDataset(
#     csv_file=val_csv_file, img_dir=val_img_dir, transform=transform
# )

# # Create a DataLoader
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# train_size = len(train_dataset)
# test_size = len(test_dataset)
# val_size = len(val_dataset)


# 4 CNN layer NN
class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 6 * 6, 1024)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)

        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


# 6 CNN layer NN
class EnhancedEmotionModel1(nn.Module):
    def __init__(self):
        super(EnhancedEmotionModel1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = self.dropout3(x)

        x = x.view(-1, 512 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 5 CNN layer NN
class EnhancedEmotionModel2(nn.Module):
    def __init__(self):
        super(EnhancedEmotionModel2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout2(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)

        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = self.dropout3(x)

        x = x.view(-1, 512 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# MobileNet Architecture
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MobileNet(nn.Module):
    def __init__(self, num_classes=7):
        super(MobileNet, self).__init__()
        self.num_classes = num_classes

        def conv_bn(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def conv_dw(in_channels, out_channels, stride):
            return nn.Sequential(
                DepthwiseSeparableConv(in_channels, out_channels, stride)
            )

        self.model = nn.Sequential(
            conv_bn(
                1, 32, 2
            ),  # Change the first layer to accept 1 channel instead of 3
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.5)  # Add dropout layer with 50% dropout rate
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.dropout(x)  # Apply dropout before the fully connected layer
        x = self.fc(x)
        return x


# Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, path="checkpoint.pt"):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# 4 layer CNN Model
# model = EmotionModel().to(device)

# 7 layer CNN model
# model = EnhancedEmotionModel1().to(device)

# 5 layer CNN model
model = EnhancedEmotionModel2().to(device)

# MobileNet Architecture
# model = MobileNet(num_classes=7).to(device)

# Using pretrained MobileNet
# model = models.mobilenet_v2(pretrained=True)

# # Modify the first convolutional layer to accept grayscale images
# model.features[0][0] = torch.nn.Conv2d(
#     1, 32, kernel_size=3, stride=2, padding=1, bias=False
# )

# # Modify the last layer to output 7 classes
# model.classifier[1] = torch.nn.Linear(model.last_channel, 7)

# model.to(device)

# # Freeze all layers initially and unfreeze the last few layers
# # Freeze all layers initially
# for param in model.parameters():
#     param.requires_grad = False

# # Unfreeze the last few layers
# for param in model.features[-5:].parameters():
#     param.requires_grad = True

# # Ensure the final classifier is also trainable
# for param in model.classifier.parameters():
#     param.requires_grad = True


# summary(model, (1, 48, 48))

# Define the optimizer and loss function
optimizer1 = optim.NAdam(model.parameters(), lr=0.00005)
optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
criterion = nn.CrossEntropyLoss()


steps_per_epoch = (train_size // batch_size) + 1
val_steps_per_epoch = (val_size // batch_size) + 1

# Training the model
num_epochs = 50
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


#########################################           Training               ##############################

# # Instantiate EarlyStopping
# early_stopping = EarlyStopping(patience=5, verbose=True)


# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0
#     train_correct = 0
#     count = 1
#     val_count = 1
#     for batch_idx, (images, labels) in enumerate(train_loader):
#         images, labels = images.to(device), labels.to(device)  # Move data to GPU

#         optimizer1.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer1.step()
#         train_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         train_correct += (predicted == labels).sum().item()

#         print(
#             f"Step Count = {count}/{steps_per_epoch}  , Epoch = {epoch+1}/{num_epochs}"
#         )
#         count += 1

#         if batch_idx % 100 == 0:
#             print(
#                 f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}"
#             )

#     train_losses.append(train_loss / len(train_loader))
#     train_accuracies.append(100 * train_correct / len(train_dataset))

#     model.eval()
#     val_loss = 0
#     val_correct = 0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)  # Move data to GPU

#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             val_correct += (predicted == labels).sum().item()

#             print(
#                 f"Val_Step Count = {val_count}/{val_steps_per_epoch}  , Epoch = {epoch+1}/{num_epochs}"
#             )
#             val_count += 1

#     val_losses.append(val_loss / len(val_loader))
#     val_accuracies.append(100 * val_correct / len(val_dataset))

#     print(
#         f"Epoch {epoch+1}/{num_epochs}, "
#         f"Train Loss: {train_losses[-1]:.4f}, "
#         f"Train Accuracy: {train_accuracies[-1]:.2f}%, "
#         f"Val Loss: {val_losses[-1]:.4f}, "
#         f"Val Accuracy: {val_accuracies[-1]:.2f}%"
#     )

#     # Check if validation loss has decreased
#     early_stopping(val_loss, model)

#     # Break the training loop if early stopping criterion is met
#     if early_stopping.early_stop:
#         print("Early stopping")
#         break

# torch.save(model.state_dict(), "model2_pytorch_10.6.pt")

##################################         Training with FERPLUS        ##############################
# import torch.nn.functional as F

# # Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Move the model to GPU if available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Training loop
# num_epochs = 10

# for epoch in range(num_epochs):  # Loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(train_loader):
#         # Get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)

#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)

#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()

#         # Print statistics
#         running_loss += loss.item()
#         if i % 100 == 99:  # Print every 100 mini-batches
#             print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}")
#             running_loss = 0.0

# print("Finished Training")


# ##################################           Metrics Plotting             ############################
# # Plotting accuracy graph
# plt.plot(train_accuracies, label="training accuracy")
# plt.plot(val_accuracies, c="red", label="validation accuracy")
# plt.xlabel("number of epochs")
# plt.ylabel("accuracy")
# plt.title("acc vs v-acc")
# plt.legend()
# plt.show()

# # Plotting loss graph
# plt.plot(train_losses, label="training loss")
# plt.plot(val_losses, c="red", label="validation loss")
# plt.xlabel("number of epoch")
# plt.ylabel("loss value")
# plt.title("loss vs v-loss")
# plt.legend()
# plt.show()

######################################              Testing              #######################################


# model = EnhancedEmotionModel2().to(device)
# model.load_state_dict(torch.load("model2_pytorch_10.6.pt"))
# summary(model, (1, 48, 48))

# model.eval()
# model.to(device)
# image_dir = (
#     "C:/Users/jishn/OneDrive/Desktop/Emotion Detection Project/Datasets/FER2013/test"
# )

# # Load datasets
# test_dataset = datasets.ImageFolder(
#     "C:/Users/jishn/OneDrive/Desktop/Emotion Detection Project/Datasets/FER2013/test",
#     transform=transform,
# )

# image_size = (48, 48)  # Update this size according to your model's input size

# transform = transforms.Compose(
#     [
#         transforms.Resize(image_size),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.5], std=[0.5]
#         ),  # Normalize with mean and std for grayscale
#     ]
# )


# def load_image(image_path):
#     image = Image.open(image_path).convert("L")  # Keep the image in grayscale
#     image = transform(image)
#     image = image.unsqueeze(0)  # Add batch dimension
#     return image


# image_paths = []
# true_labels = []

# test_count = 1
# for class_index, class_name in enumerate(sorted(os.listdir(image_dir))):
#     class_folder = os.path.join(image_dir, class_name)
#     if os.path.isdir(class_folder):
#         for filename in os.listdir(class_folder):
#             if filename.endswith(".png") or filename.endswith(
#                 ".jpg"
#             ):  # Adjust as needed
#                 image_path = os.path.join(class_folder, filename)
#                 image_paths.append(image_path)
#                 true_labels.append(class_index)
#                 print(f"Test Images Proecssed: {test_count}/{len(test_dataset)}")
#                 test_count += 1

# images = []
# for image_path in image_paths:
#     images.append(load_image(image_path))

# images = torch.cat(images).to(device)  # Combine into a single tensor and move to GPU
# true_labels_tensor = torch.tensor(true_labels).to(
#     device
# )  # Convert true labels to a tensor and move to GPU

# # Define loss function
# criterion = nn.CrossEntropyLoss()

# # Make predictions
# with torch.no_grad():
#     outputs = model(images)
#     _, predicted_classes = torch.max(outputs, 1)

# # Calculate loss
# loss = criterion(outputs, true_labels_tensor)
# print(f"Loss: {loss.item()}")

# # Calculate accuracy
# correct_predictions = (predicted_classes == true_labels_tensor).sum().item()
# accuracy = correct_predictions / len(true_labels)
# print(f"Accuracy: {accuracy * 100:.2f}%")

# # Print confusion matrix
# cm = confusion_matrix(true_labels, predicted_classes.cpu().numpy())
# cm_display = ConfusionMatrixDisplay(cm, display_labels=sorted(os.listdir(image_dir)))
# cm_display.plot()
# plt.show()


####################################################  LIVE FEED TESTING 1 ####################################################

import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# Load your trained model
# model = EmotionModel().to(device)
# model.load_state_dict(torch.load("model2_pytorch_8.6.pt"))
# model.eval()

# Define preprocessing transformations
# preprocess = transforms.Compose(
#     [
#         transforms.Resize((48, 48)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.5], std=[0.5]
#         ),  # Normalize with mean and std for grayscale
#     ]
# )

# # Initialize the webcam
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert the image from BGR to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Convert the image to a PIL image
#     pil_image = Image.fromarray(rgb_frame)

#     # Preprocess the image
#     input_tensor = preprocess(pil_image)
#     input_tensor = input_tensor.unsqueeze(
#         0
#     )  # Create a mini-batch as expected by the model

#     # Move the input tensor to the appropriate device
#     input_tensor = input_tensor.to("cuda" if torch.cuda.is_available() else "cpu")

#     # Run the model on the input tensor
#     with torch.no_grad():
#         output = model(input_tensor)

#     # Process the model output
#     _, predicted = torch.max(output, 1)
#     label = predicted.item()  # This should be the index of the class

#     # Display the label on the frame
#     cv2.putText(
#         frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
#     )

#     # Display the resulting frame
#     cv2.imshow("Live Camera Feed", frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # When everything is done, release the capture
# cap.release()
# cv2.destroyAllWindows()


####################################################  LIVE FEED TESTING 2 ####################################################

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


# warnings.filterwarnings("ignore")

# # Load your trained PyTorch model
# model = EmotionModel().to(device)
# model.load_state_dict(torch.load("model2_pytorch_8.6.pt"))

# model.eval()
# summary(model, (1, 48, 48))

# # Initialize face detector
# face_haar_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# # Define preprocessing transformations
# preprocess = transforms.Compose(
#     [
#         transforms.Resize((48, 48)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.5], std=[0.5]
#         ),  # Normalize with mean and std for grayscale
#     ]
# )

# # Initialize the webcam
# cap = cv2.VideoCapture(0)

# emotions = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

# while True:
#     ret, test_img = (
#         cap.read()
#     )  # captures frame and returns boolean value and captured image
#     if not ret:
#         continue
#     gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

#     faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

#     for x, y, w, h in faces_detected:
#         cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
#         roi_gray = gray_img[
#             y : y + w, x : x + h
#         ]  # cropping region of interest i.e. face area from image
#         pil_image = Image.fromarray(roi_gray)

#         # Preprocess the image
#         input_tensor = preprocess(pil_image)
#         input_tensor = input_tensor.unsqueeze(
#             0
#         )  # Create a mini-batch as expected by the model

#         # Move the input tensor to the appropriate device
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         input_tensor = input_tensor.to(device)
#         model.to(device)

#         # Run the model on the input tensor
#         with torch.no_grad():
#             output = model(input_tensor)

#         # Process the model output
#         max_index = torch.argmax(output[0]).item()
#         predicted_emotion = emotions[max_index]

#         cv2.putText(
#             test_img,
#             predicted_emotion,
#             (int(x), int(y)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 0, 255),
#             2,
#         )

#     resized_img = cv2.resize(test_img, (1000, 700))
#     cv2.imshow("Facial emotion analysis", resized_img)

#     if cv2.waitKey(10) == ord("q"):  # wait until 'q' key is pressed
#         break

# cap.release()
# cv2.destroyAllWindows()


########################## FLASK APP ######################################
# from flask import Flask, render_template, Response
# import cv2

# app = Flask(__name__)


# def gen_frames():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     warnings.filterwarnings("ignore")

#     # Load your trained PyTorch model
#     new_model = EmotionModel().to(device)
#     new_model.load_state_dict(torch.load("model1_pytorch_8.6.pt"))
#     model.eval()

#     # Initialize face detector
#     face_haar_cascade = cv2.CascadeClassifier(
#         cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#     )

#     # Define preprocessing transformations
#     preprocess = transforms.Compose(
#         [
#             transforms.Resize((48, 48)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.5], std=[0.5]
#             ),  # Normalize with mean and std for grayscale
#         ]
#     )

#     # Initialize the webcam
#     cap = cv2.VideoCapture(0)

#     emotions = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

#     while True:
#         ret, test_img = cap.read()
#         if not ret:
#             continue
#         gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

#         faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

#         for x, y, w, h in faces_detected:
#             cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
#             roi_gray = gray_img[
#                 y : y + w, x : x + h
#             ]  # cropping region of interest i.e. face area from image
#             pil_image = Image.fromarray(roi_gray)

#             # Preprocess the image
#             input_tensor = preprocess(pil_image)
#             input_tensor = input_tensor.unsqueeze(
#                 0
#             )  # Create a mini-batch as expected by the model

#             # Move the input tensor to the appropriate device
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             input_tensor = input_tensor.to(device)
#             model.to(device)

#             # Run the model on the input tensor
#             with torch.no_grad():
#                 output = model(input_tensor)

#             # Process the model output
#             max_index = torch.argmax(output[0]).item()
#             predicted_emotion = emotions[max_index]

#             cv2.putText(
#                 test_img,
#                 predicted_emotion,
#                 (int(x), int(y)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 0, 255),
#                 2,
#             )

#         resized_img = cv2.resize(test_img, (1000, 700))
#         ret, buffer = cv2.imencode(".jpg", resized_img)
#         frame = buffer.tobytes()
#         yield (
#             b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
#         )  # Concatenate frame one by one and show result

#         # cv2.imshow("Facial emotion analysis", resized_img)

#     # camera = cv2.VideoCapture(0)  # Use 0 for web camera
#     # while True:
#     #     success, frame = camera.read()  # Read the camera frame
#     #     if not success:
#     #         break
#     #     else:
#     #         ret, buffer = cv2.imencode(".jpg", frame)
#     #         frame = buffer.tobytes()
#     #         yield (
#     #             b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
#     #         )  # Concatenate frame one by one and show result


# @app.route("/")
# def index():
#     return render_template("index1.html")


# @app.route("/video_feed")
# def video_feed():
#     return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# if __name__ == "__main__":
#     app.run(debug=True)

############################ UPLOAD PHOTO TEST ######################################


# import os
# from flask import Flask, request, render_template, redirect, url_for, jsonify
# from werkzeug.utils import secure_filename
# import torch
# from torch import nn
# from torchvision import transforms, datasets
# from PIL import Image
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# app = Flask(__name__)
# app.config["UPLOAD_FOLDER"] = "uploads"
# app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

# # Ensure upload folder exists
# os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# # Load the model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = EmotionModel().to(device)
# model.load_state_dict(torch.load("model2_pytorch_8.6.pt"))
# model.eval()

# # Define image transformations
# image_size = (48, 48)  # Update this size according to your model's input size
# transform = transforms.Compose(
#     [
#         transforms.Resize(image_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize for grayscale images
#     ]
# )


# def allowed_file(filename):
#     return (
#         "." in filename
#         and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
#     )


# def load_image(image_path):
#     image = Image.open(image_path).convert("L")  # Keep the image in grayscale
#     image = transform(image)
#     image = image.unsqueeze(0)  # Add batch dimension
#     return image.to(device)


# @app.route("/")
# def index():
#     return render_template("index2.html")


# @app.route("/upload", methods=["POST"])
# def upload_file():
#     if "file" not in request.files:
#         return redirect(request.url)
#     file = request.files["file"]
#     if file.filename == "":
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#         file.save(filepath)

#         image = load_image(filepath)
#         with torch.no_grad():
#             output = model(image)
#             _, predicted_class = torch.max(output, 1)
#             predicted_class = predicted_class.item()

#         # Assuming you have a list of class names
#         class_names = sorted(
#             os.listdir(
#                 "C:/Users/jishn/OneDrive/Desktop/Emotion Detection Project/Datasets/FER2013/test"
#             )
#         )
#         predicted_label = class_names[predicted_class]

#         return jsonify({"predicted_label": predicted_label})


# if __name__ == "__main__":
#     app.run(debug=True)

########################################      LIVE FEED TEST IN FLASK     #############################################
import os
from flask import Flask, render_template, Response
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedEmotionModel2().to(device)
model.load_state_dict(torch.load("model2_pytorch_10.6.pt"))
model.eval()

# Define image transformations
image_size = (48, 48)  # Update this size according to your model's input size
transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize for grayscale images
    ]
)

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def load_image(image):
    image = Image.fromarray(image).convert("L")  # Keep the image in grayscale
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)


def gen_frames():
    camera = cv2.VideoCapture(0)  # Use 0 for web camera
    while True:
        success, frame = camera.read()  # Read the camera frame
        if not success:
            break
        else:
            # Convert the frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            for x, y, w, h in faces:
                # Draw bounding box around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Extract the face region
                face_region = frame[y : y + h, x : x + w]

                # Resize and transform the face region for the model
                face_region_resized = cv2.resize(face_region, (48, 48))
                image_tensor = load_image(face_region_resized)

                # Make prediction
                with torch.no_grad():
                    output = model(image_tensor)
                    _, predicted_class = torch.max(output, 1)
                    predicted_class = predicted_class.item()

                # Assuming you have a list of class names
                class_names = sorted(
                    os.listdir(
                        "C:/Users/jishn/OneDrive/Desktop/Emotion Detection Project/Datasets/FER2013/test"
                    )
                )
                predicted_label = class_names[predicted_class]

                # Display the prediction label above the bounding box
                cv2.putText(
                    frame,
                    predicted_label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )

            # Encode frame as JPEG
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )  # Concatenate frame one by one and show result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)
