import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.transforms as transforms
import torch.nn.functional as func

print(torch.cuda.is_available())
print(torch.__version__)


train_set= torchvision.datasets.FashionMNIST(
    train= True,
    root= '../data/FashionMNIST/train/',
    download= True,
    transform= transforms.Compose([transforms.ToTensor()])
)
test_set= torchvision.datasets.FashionMNIST(
    train= False,
    root= '../data/FashionMNIST/test/',
    download= True,
    transform= transforms.Compose([transforms.ToTensor()])
)
print("train set:", len(train_set))
print("test set:", len(test_set))



# Network
# formula for channels 

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Convolutional layer1
        self.conv1= nn.Sequential()
        self.conv1.add_module("conv1", nn.Conv2d(in_channels= 1, out_channels= 6, kernel_size= 5))
        self.conv1.add_module("bn1", nn.BatchNorm2d(num_features= 6, eps= 1e-05, momentum= 0.1, affine= True))
        self.conv1.add_module("relu", nn.ReLU(inplace= False))
        self.conv1.add_module("pool", nn.MaxPool2d(kernel_size= 2, stride= 2))

        # Convolutional layer2
        self.conv2= nn.Sequential()
        self.conv2.add_module("conv2", nn.Conv2d(in_channels= 6, out_channels= 12, kernel_size= 5))
        self.conv2.add_module("bn2", nn.BatchNorm2d(num_features= 12, eps= 1e-05, momentum= 0.1, affine= True))
        self.conv2.add_module("relu", nn.ReLU(inplace= False))
        self.conv2.add_module("pool", nn.MaxPool2d(kernel_size= 2, stride= 2))

        # Linear layer1
        self.fc1= nn.Sequential()
        self.fc1.add_module("linear", nn.Linear(in_features= 12*4*4, out_features= 120))
        #  self.fc1.add_module("bn3", nn.BatchNorm1d(num_features= 120, eps= 1e-05, momentum= 0.1, affine= True))
        self.fc1.add_module("relu", nn.ReLU(inplace= False))
        
        # Linear layer2
        self.fc2= nn.Sequential()
        self.fc2.add_module("linear", nn.Linear(in_features= 120, out_features= 60))
        #self.fc2.add_module("bn4", nn.BatchNorm1d(num_features= 60, eps= 1e-5, momentum= 0.1, affine= True))
        self.fc2.add_module("relu", nn.ReLU(inplace= False))

        # Output layer
        self.out= nn.Sequential()
        self.out.add_module("output", nn.Linear(in_features= 60, out_features= 10))
        # by default activation function is softmax

    def forward(self, t):
        t= t
        t= self.conv1(t)
        t= self.conv2(t)
        t= t.reshape(-1, 12*4*4)
        t= self.fc1(t)
        t= self.fc2(t)
        t= self.out(t)
        return t


# train test epochs
device= torch.device('cuda')
network= Network()
train_loader= torch.utils.data.DataLoader(train_set, batch_size= 5000)
test_loader= torch.utils.data.DataLoader(test_set, batch_size= 5000)
optimizer= optim.Adam(network.parameters(), lr= 0.1)
test_batch= next(iter(test_loader))

# gpu specific
    network.cuda()

epochs= 200
for epoch in range(1, epochs):
    i= 0
    for batch in train_loader:
        i+=1
        images, labels= batch
        images= images.to('cuda')
        labels= labels.to('cuda')
        if epoch == 1 and i == 1:
            #First time check and resume checkpoint
            try:
                print('resuming')
                checkpoint = torch.load('model/checkpoint')
                network.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                network.train()
                print('epoch', epoch)
            except:
                pass
            
        # .to(cuda) for GPU

        preds= network(images)
        train_loss= func.cross_entropy(preds, labels)

        optimizer.zero_grad()
        train_loss.backward(retain_graph=True)
        optimizer.step()
        
        images, labels= test_batch
        # .to(cuda) for GPU
        images= images.to('cuda')
        labels= labels.to('cuda')
        preds= network(images)
        test_loss= func.cross_entropy(preds, labels)

        # checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
            }, 'model/checkpoint')

        print('training on epoch: {} and batch: {}, train loss: {:.2f}, test loss: {:.2f}'.format(epoch, i, train_loss.item(), test_loss.item()))
        PATH= 'model/checkpoint_{:.2f}'.format(test_loss.item())
        torch.save(network.state_dict(), PATH)



# final testing:
def get_num_correct(preds, labels):
    return torch.argmax(preds, dim= 1).eq(labels).sum()

network= network.eval()
test_loader= torch.utils.data.DataLoader(test_set, batch_size= 10000)
images, labels= test_batch
images= images.to('cuda')
labels= labels.to('cuda')
preds= network(images)
print('correctly classified from 10000 images:', get_num_correct(preds, labels).item())

