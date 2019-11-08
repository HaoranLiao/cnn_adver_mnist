import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import data as datagen

def load_train_images():
    (train_data, val_data, test_data) = datagen.get_data_web([6,7], 0, [8,8], 2, sample_size=1000)
                
    (train_images, train_labels) = train_data
    train_images = np.reshape(train_images, [1000, 1, 8, 8])
    train_images = torch.from_numpy(train_images).to(dtype=torch.float32)
    train_labels = torch.from_numpy(train_labels).to(dtype=torch.int64)
         
    (test_images, test_labels) = test_data
    test_images = np.reshape(test_images, [1000, 1, 8, 8])
    test_images = torch.from_numpy(test_images).to(dtype=torch.float32)
    test_labels = torch.from_numpy(test_labels).to(dtype=torch.int64)
    
    return (train_images, train_labels, test_images, test_labels)
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv1_drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(20, 40, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(40)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(40, 80, kernel_size=2) 
        self.bn3 = nn.BatchNorm2d(80)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 80)
        self.fc2 = nn.Linear(80, 10)
        self.fc3 = nn.Linear(10, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): #[samp, 1, 8, 8]
        x = self.conv1(x) #[samp, 20, 7, 7]
        self.bn1(x)
        self.conv1_drop(x)
        x = self.conv2(x) #[samp, 40, 6, 6]
        self.bn2(x)
        self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2)) #[samp, 80, 3, 3]
        x = self.conv3(x) #[samp, 80, 2, 2]
        self.bn3(x)
        self.conv3_drop(x) #[samp, 80, 2, 2]
        x = F.relu(x)
        x = x.view(-1, 320) #[samp, 320]
        x = self.fc1(x) #[samp, 80]
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x)) #[samp, 10]
        x = F.dropout(x, training=self.training)
        x = self.fc3(x) #[samp, 2]
        x = self.sigmoid(x)
        return x
    
(train_images, train_labels, test_images, test_labels) = load_train_images()
network = Net()
optimizer = optim.Adam(network.parameters())

def train():
  network.train()
  batch_idx = 0
  batch_iter_train = datagen.batch_generator(train_images, train_labels, 100)
  for (data, target) in batch_iter_train:
      batch_idx += 1
      optimizer.zero_grad()
      output = network(data)
      loss = ((output - target)**2).mean()
      loss.backward()
      optimizer.step()
      
def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    batch_iter_test = datagen.batch_generator(test_images, test_labels, 1000)
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

print('Test Accuracy: %.3f'%test())
for epoch in range(1, 101):
  print('Epoch: %s'%epoch)
  train()
  print('Test Accuracy: %.3f'%test())
  
#torch.save(network, './trained/samp1000_size8_dig67.pth')
#print('Model saved')