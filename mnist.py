import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size= 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size= 5)
        self.dropout = nn.Dropout2d()
        
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100,10)   
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(F.dropout2d(self.conv2(x)), 2))
        
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

epochs = 5
batch_sz = 100
learning_rate = 0.001

train_dataset = dsets.MNIST(root = './data', 
                             train= True,
                             download=True, 
                             transform= transforms.ToTensor())

test_dataset = dsets.MNIST(root= './data', 
                            train = False,
                            transform= transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(batch_size= batch_sz,
                                           dataset= train_dataset, 
                                           shuffle= True)

test_loader = torch.utils.data.DataLoader(batch_size= batch_sz, 
                                          dataset = test_dataset, 
                                          shuffle = False)

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        
        if (i+1) % batch_sz == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                 %(epoch+1, epochs, i+1, len(train_dataset)//batch_sz, loss.data[0]))

i=0
correct = 0
total = 0
for images, labels in (test_loader):
    images = Variable(images)
    output = net(images)
    _, pred = torch.max(output.data, 1)
    correct += (labels == pred).sum().item()
    total += len(pred)
print('Accuracy of the network on the 10K test images: %.2f %%' % (100 * correct / float(total) ))