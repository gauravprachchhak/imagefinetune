import torch
import torchvision.models as models
import torch.nn as nn
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
NUM_EPOCH = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResNet50_CIFAR(nn.Module):
    def __init__(self):
        super(ResNet50_CIFAR, self).__init__()
        # Initialize ResNet 50 with ImageNet weights
        ResNet50 = models.resnet50(pretrained=True)
        modules = list(ResNet50.children())[:-1]
        backbone = nn.Sequential(*modules)
        # Create new layers
        self.backbone = nn.Sequential(*modules)
        self.fc1 = nn.Linear(2048, 32)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, img):
        # Get the flattened vector from the backbone of resnet50
        out = self.backbone(img)
        # processing the vector with the added new layers
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        return self.fc2(out)

def train():
    ## Define the training dataloader
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10('./data', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

    ## Create model, objective function and optimizer
    model = ResNet50_CIFAR()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.fc1.parameters()) + list(model.fc2.parameters()),
                           lr=0.001, momentum=0.9)

    ## Do the training
    for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
    print('Finished Training')

    # save the model
    torch.save(model,'trained.pth')





if __name__ == '__main__':
    train()
