import torch
import torchvision
from torchvision.transforms import transforms
from torch import nn

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


test_set= torchvision.datasets.FashionMNIST(
    train= False,
    root= '../data/FashionMNIST/test/',
    download= True,
    transform= transforms.Compose([transforms.ToTensor()])
)

PATH= 'model/checkpoint'

print(PATH)
model= Network()
device = torch.device("cpu")
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model= model.eval()

test_loader= torch.utils.data.DataLoader(test_set, batch_size= 1)
batch= next(iter(test_loader))
images, labels= batch
pred= model(images)
print(torch.argmax(pred), labels.item())
