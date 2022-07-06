

# here we import the libraries that we will use
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# you should type in your path here to be able to work with the code
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,         
                                        download=True, transform=transform) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2,
                                          shuffle=True, num_workers=2)

# type in the same path as you typed above
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)      
testloader = torch.utils.data.DataLoader(testset, batch_size=2,
                                         shuffle=False, num_workers=2)

# here we categorize the classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# here we import additional libraries
import matplotlib.pyplot as plt
import numpy as np

# we define the functions to show the images
def imshow(img):
# here we unnormalize 
    img = img / 2 + 0.5    
    npimg = img.numpy()
    # we plot the first graph
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # here we print the first graph
    plt.show()


# here we get some random images to be able to train 
dataiter = iter(trainloader)
images, labels = dataiter.next()

#  we show the images 
imshow(torchvision.utils.make_grid(images))
# we print labels which are in range of 2
print(' '.join('%5s' % classes[labels[j]] for j in range(2)))

# again we import some other libraries
import torch.nn as nn 
import torch.nn.functional as F


class Net(nn.Module):
    # we use this initialize the model
    def __init__(self):
        super(Net, self).__init__()
        # We apply a 2d convolution over the input which is first layer of NN.The params: # of channels in, # of channels out and finally the filter size 
        self.conv1 = nn.Conv2d(3, 6, 5)
         # Here we apply a 2d max pooling over the input. Which has the following params: Height and width of filter size and stride
        self.pool = nn.MaxPool2d(2, 2) 
         # Now we do 2d convolution for the second layer of the NN
        self.conv2 = nn.Conv2d(6, 16, 5)
         # This applies a linear transformation to the input data. The parameters are input and output features
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Linear transformation for layer 2
        self.fc2 = nn.Linear(120, 84)
        #Linear transformation for the output layer. We use 10 because there are 10 possible output classes.
        self.fc3 = nn.Linear(84, 10) 
    
    # this is used to predict
    def forward(self, x):
        # x initial image pass through conv1 after that it goes through relu activation function and at last to Max pool
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))
        # here we flatten like we did before
        x = x.view(-1, 16 * 5 * 5)
        #fc 1 is passed flattened image 
        x = F.relu(self.fc1(x))
        #fc 2 is passed as well
        x = F.relu(self.fc2(x))
        # here we predict 10 values each for one class 
        x = self.fc3(x)
        # we finally end with returning x
        return x

# here we init the network using the following
net = Net()

# here we import another class which will be used 
import torch.optim as optim

# cross-entropy loss is put into criterion
criterion = nn.CrossEntropyLoss()
# Optimizer is set to the SGD, we've set lr to be 0.001 and momentum to be 0.9
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# now we want to train
# Loop over the dataset
for epoch in range(100): 
 # We initate running_loss to zero to be able to keep track of our loss
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs: data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward and backward propagation, and optimization
        outputs = net(inputs)
        # we want to calculate loss using predicted outputs and gt
        loss = criterion(outputs, labels)
        # here we backpropagate the loss
        loss.backward()
        # later we realize gradient descent, and finally update the parameters
        optimizer.step()

        # here we put the statistics to the console
        running_loss += loss.item()
        # here we check if the condition is true and if it is we print every mini-batches (equal to priting every 2000 mini batches)
        if i % 2000 == 1999:    
             #We print the epoch, the mini-batch its on, and its loss at that point
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            #running_loss is made 0.0 again
            running_loss = 0.0

#we define the path and save all the params to the path that we just set
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH) 

# now we test the network with the testing data
# This will iterate through the test data to check if we could make the model learndataiter = iter(testloader)
images, labels = dataiter.next()

# Here print images taken from testloader
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(2)))
# This will print 2 images from the test with theur correct labels 

# we again init the network like we did before
net = Net()
# we load weights
net.load_state_dict(torch.load(PATH)) 
# We put these four images into the model
outputs = net(images)
# This will give a prediction from model
_, predicted = torch.max(outputs, 1) 

# We print the predictions. The outputs we get are weights of the ten classes.
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(2)))

# Now we will test the whole dataset

# Here we intiate correct and total as zero. We will use them to keep track of our progress.
correct = 0
total = 0
with torch.no_grad():
    #Here we run every image in the test dataset. We are passing every single image.
    for data in testloader:
        images, labels = data
        outputs = net(images)
        # We count the total amount and correct amount to keep track of acciracy.
        _, predicted = torch.max(outputs.data, 1)
        # here total becomez labels.size(0)
        total += labels.size(0)
        correct += (predicted == labels).sum().item() 

# Then we print the accuracy we got from testing 10000 images.
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# the following is used to get accuracy of each class
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(2):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

#prints accuracy for each class
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

