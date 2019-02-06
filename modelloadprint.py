'''
Code for Testloader for generating HTML
'''
import torch
import torchvision.models as models
import torch.nn as nn
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as im
import numpy as np

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

def test():
    #Declare batch size for seperating images
    batch_size = 4
    #Load trained model and map it explicitly to cpu as it was trained on gpu
    trained_model = torch.load('trained.pth',map_location='cpu')
    #Create transform same as it was during training
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #Define testset from CIFAR10 dataset
    testset = datasets.CIFAR10('./data', download=True, train=False, transform=transform)
    #Create testloader that loads the testset with no shuffle
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    #Define classess for easy understanding and printing purpose
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    #Create HTML 
    html = """
    <html>
    <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
    <table border = '1' align='center' width='100%'>
    <caption>Test images with their classification scores in terms of softmax probabilites</caption>
    <th>Image</th>
    <th>Classification Scores</th>
    <th>Real</th>
    <th>Predicted</th>
    """
    
    #See if directory of images is there else create one
    if not os.path.exists('images'):
        os.makedirs('images')
        
    #Loop through all datapoints of testloader
    for i,data in enumerate(testloader, 0):
        #Break the loop if i>10 for simplicity purpose
        #If you want to run this for all 10000 images then remove this condition
        if i > 10:
            break
        #Fetch images and labels from data
        images, labels = data
        #Fetch cross entropy scores from trained models
        outputs = trained_model(images)
        #Generate predicted values
        _, predicted = torch.max(outputs,1)
        #Generate Softmax object for converting cross entropy scores into probabilities
        sm = torch.nn.Softmax(dim=1)
        #Compute Softmax on outputs
        probabilities = sm(outputs) 
        #For loop to seperate images
        for j in range(batch_size):
            #Fetch jth index of probabilities and assign to classscr
            classscr = probabilities[j]
            #Fetch jth index of images and assign it to imgarr
            imgarr = images[j]
            #Normalize
            img = imgarr / 2 + 0.5
            #Convert img to numpy
            image = img.numpy()
            #Define path for images to save
            path = 'images/'+str(j)+str(i)+'.jpeg'
            #Save images in the mentioned path
            im.imsave(path,np.transpose(image, (1, 2, 0)))
            #Put thr saved image in image tag of HTML
            html += "<tr><td align='center'><img src ={0}></td><td>".format(path)
            #For loop for extracting 10 probabilities values
            for k in range(0,10):
                #Logic for creating progress bars according to their percentage probabilities
                if classes[labels[j]] == classes[predicted[j]]:
                    if max(classscr) == classscr[k]:
                        html += "<div style='width:{1}%;background-color:lightgreen;'><b>{0}</b>:{2:.2f}</div><br/>".format(classes[k],classscr[k]*100,classscr[k])
                    else:
                        html += "<div style='width:{1}%;background-color:lightblue;'><b>{0}</b>:{2:.2f}</div><br/>".format(classes[k],classscr[k]*100,classscr[k])
                else:
                    html += "<div style='width:{1}%;background-color:lightblue;'><b>{0}</b>:{2:.2f}</div><br/>".format(classes[k],classscr[k]*100,classscr[k])
            #Put the real class of the image in HTML
            html += "</td><td align='center'>{0}</td>".format(classes[labels[j]])
            #Put the predicted class of the image in HTML
            html += "<td align='center'>{0}</td></tr>".format(classes[predicted[j]])
    #Close HTML
    html += "</table></html>"
    #Open file to write HTML 
    f = open('imagefinetune.html','w')
    #Write HTML
    f.write(html)
    
 
if __name__ == '__main__':
    test()