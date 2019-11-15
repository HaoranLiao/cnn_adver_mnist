import torch
import numpy as np
import ntpath, re
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from package import train_cnn_size8
from package import train_cnn_size16

model = train_cnn_size16.Net()
model.load_state_dict(torch.load('../trained_models/samp2500_size16_dig67.pth'))
model.eval()

inputfile = '../adver_attack/size16_noisy_grads/pair67_dig6_ind3_size16.npy'
inputdigit_smaller = []

#7_ind5, 7_ind12
#epsilons = [0, 0.1, 0.15,0.20,0.22, .23,0.24, 0.25,.26, 0.30,.4,.5]
#6_ind3
epsilons = [0, 0.1, 0.15,.16,.17,.18,.19,0.20,0.22,0.24,.26, 0.30,.4,.5]


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def load_test_image(inputfile):
    test_image = np.load(inputfile)
    test_image = test_image[None, None, ...]
    tmp = path_leaf(inputfile)
    test_label_int = int(re.search('dig(.+?)_ind', tmp).group(1))
    test_pair_str = re.search('pair(.+?)_dig', tmp).group(1)
    if test_label_int == int(test_pair_str[0]):
        test_label = np.array([[1, 0]])
        inputdigit_smaller.append(1)
    else:
        test_label = np.array([[0, 1]])
        inputdigit_smaller.append(0)
        
    test_image = torch.from_numpy(test_image).to(dtype=torch.float32)
    test_label = torch.from_numpy(test_label).to(dtype=torch.int64)
    
    yield test_image, test_label

def save_image(image, inputfile, epsilon, confidence, acc):
    image = np.squeeze(image.detach().numpy())
    plt.figure()
    plt.imshow(image)
    plt.title(
        str(epsilon)+'_'+str(confidence)+'_'+'%.2f'%acc)
    outfn = inputfile[0:-4]+'_att%.2f_noisy'%epsilon
    plt.savefig(outfn+'.png')

def fgsm_attack_noisy_grads(image, epsilon, data_grad):
    noise = torch.from_numpy(
            np.random.normal(
            loc=0,
            scale=1e-5,
            size=(data_grad.numpy().shape[0], 
                  data_grad.numpy().shape[1], 
                  data_grad.numpy().shape[2],
                  data_grad.numpy().shape[3])
            )
    ).to(dtype=torch.float32)
    data_grad = data_grad + noise
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image

def test(model, test_loader, epsilon):
    correct = 0
    confidence = []
    adv_examples = []
    for data, target in test_loader:
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        init_pred_int = [6 if x == 0 else 7 for x in init_pred[0]][0]
        if init_pred.item() != target.max(1, keepdim=True)[1].item():
            continue
        
        loss = ((output - target)**2).mean()
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack_noisy_grads(data, epsilon, data_grad)
        pert_output = model(perturbed_data)
        
        confidence.append(list(pert_output.detach().numpy()[0]))
        pert_pred = pert_output.max(1, keepdim=True)[1]
        pert_pred_int = [6 if x == 0 else 7 for x in pert_pred[0]][0]

        if pert_pred.item() == target.max(1, keepdim=True)[1].item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred_int, pert_pred_int, adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred_int, pert_pred_int, adv_ex))
        
        save_image(perturbed_data, inputfile, epsilon, confidence, correct)

    final_acc = correct/float(1)
    print("Epsilon: {}\tTest Accuracy = {}\t Confidence: {}".format(epsilon, final_acc, confidence))

    return final_acc, adv_examples, confidence

accuracies, examples, confidence = [], [], []

for eps in epsilons:
    test_loader = load_test_image(inputfile)
    acc, ex, conf = test(model, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)
    confidence.append(conf)

confidence = np.squeeze(np.array(confidence))
plt.figure(figsize=(5,5))
if inputdigit_smaller[0] == 0:
    plt.plot(epsilons, confidence[:,1], "*-")
else:
    plt.plot(epsilons, confidence[:,0], "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.ylim([0,1.0])
plt.xticks(np.arange(0, np.max(epsilons)+0.05, step=0.05))
plt.axhline(y=0.5, color='r', linestyle='-', linewidth=0.8)
plt.title(inputfile)
plt.xlabel("Epsilon")
plt.ylabel("Confidence")
plt.savefig(inputfile[0:-4]+'_noisy_conf.png')

    