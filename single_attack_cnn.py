import torch
import numpy as np
import ntpath, re

model = torch.load('./trained/samp1000_size8_dig67.pth')
model.eval()

epsilons = [0.13, 0.14, 0.15, 0.16]


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
    else:
        test_label = np.array([[0, 1]])
        
    test_image = torch.from_numpy(test_image).to(dtype=torch.float32)
    test_label = torch.from_numpy(test_label).to(dtype=torch.int64)
    
    yield test_image, test_label


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def test(model, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    confidence = []
    adv_examples = []
    # Loop over all examples in test set
    for data, target in test_loader:
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        init_pred_int = [6 if x == 0 else 7 for x in init_pred[0]][0]
        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.max(1, keepdim=True)[1].item():
            continue
        
        loss = ((output - target)**2).mean()
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        pert_output = model(perturbed_data)
        # Check for success
        confidence.append(list(pert_output.detach().numpy()[0]))
        pert_pred = pert_output.max(1, keepdim=True)[1] # get the index of the max log-probability
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

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(1)
    print("Epsilon: {}\tTest Accuracy = {}\t Confidence: {}".format(epsilon, final_acc, confidence))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples, confidence

for eps in epsilons:
    test_loader = load_test_image('./pair67_dig7_ind5.npy')
    acc, ex, conf = test(model, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)