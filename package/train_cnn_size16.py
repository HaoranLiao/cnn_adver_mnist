import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
sys.path.append('../')
from package import data as datagen

sample_size = 2500

def load_train_images():
    (train_data, val_data, test_data) = datagen.get_data_web([6,7], 0, [16,16], 2, sample_size=sample_size)
                
    (train_images, train_labels) = train_data
    train_images = np.reshape(train_images, [sample_size, 1, 16, 16])
    train_images = torch.from_numpy(train_images).to(dtype=torch.float32)
    train_labels = torch.from_numpy(train_labels).to(dtype=torch.int64)
         
    (test_images, test_labels) = test_data
    test_images = np.reshape(test_images, [test_images.shape[0], 1, 16, 16])
    test_images = torch.from_numpy(test_images).to(dtype=torch.float32)
    test_labels = torch.from_numpy(test_labels).to(dtype=torch.int64)
    
    return (train_images, train_labels, test_images, test_labels)
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=4)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv1_drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(20, 40, kernel_size=4)
        self.bn2 = nn.BatchNorm2d(40)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(40, 80, kernel_size=3) 
        self.bn3 = nn.BatchNorm2d(80)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(720, 160)
        self.fc2 = nn.Linear(160, 40)
        self.fc3 = nn.Linear(40, 10)
        self.fc4 = nn.Linear(10, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): #[samp, 1, 16, 16]
        x = self.conv1(x) #[samp, 20, 13, 13]
        self.bn1(x)
        self.conv1_drop(x)
        x = self.conv2(x) #[samp, 40, 10, 10]
        self.bn2(x)
        self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2)) #[samp, 80, 5, 5]
        x = self.conv3(x) #[samp, 80, 3, 3]
        self.bn3(x)
        self.conv3_drop(x)
        x = F.relu(x)
        x = x.view(-1, 720) #[samp, 720]
        x = self.fc1(x) #[samp, 160]
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x)) #[samp, 10]
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x)) #[samp, 2]
        x = F.dropout(x, training=self.training)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x
    
def train(network, optimizer, train_images, train_labels):
  network.train()
  batch_idx = 0
  batch_iter_train = datagen.batch_generator(train_images, train_labels, 500)
  for (data, target) in batch_iter_train:
      batch_idx += 1
      optimizer.zero_grad()
      output = network(data)
      loss = ((output - target)**2).mean()
      loss.backward()
      optimizer.step()
      
def test(network, test_images, test_labels):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    batch_iter_test = datagen.batch_generator(test_images, test_labels, 100000)
    for data, target in batch_iter_test:
      output = network(data)
      test_loss += ((output - target)**2).mean()
      correct += get_accuracy(output, target)

  return correct

def get_accuracy(output, target):
    output_index = np.argmax(output, axis=1)
    target_index = np.argmax(target, axis=1)
    compare = output_index - target_index
    compare = compare.numpy()
    num_correct = float(np.sum(compare == 0))
    total = float(output_index.shape[0])
    accuracy = num_correct / total
    return accuracy

def main():
    (train_images, train_labels, test_images, test_labels) = load_train_images()
    network = Net()
    optimizer = optim.Adam(network.parameters())
    
    print('Test Accuracy: %.3f'%test(network, test_images, test_labels))
    for epoch in range(1, 51):
      print('Epoch: %s'%epoch)
      train(network, optimizer, train_images, train_labels)
      print('Test Accuracy: %.3f'%test(network, test_images, test_labels))
      
    torch.save(network.state_dict(), '../trained_models/samp2500_size16_dig67.pth')
    print('Model saved')

  
if __name__ == "__main__":
    main()