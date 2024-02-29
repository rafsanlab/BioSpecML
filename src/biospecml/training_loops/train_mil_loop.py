from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

def train_mil_model(model, data_loader, device, num_epochs, criterion, optimizer,
                  savedir=None, f1_score_average='macro', validation_mode=False,
                  use_instance_labels=True, use_bag_labels=False,
                  one_epoch_mode=False):
    """
    Accept bag data (Bags, N-instances, Features) and do prediction based on 
    bag labels data either (Bags-labels, N-instances) or (Bags-labels).
    
    """

    metrics = {'epoch':[], 'loss':[], 'accuracy':[],'f1':[]}

    # if this function is in another running loop, set one_epoch_mode to True
    # so regardless any epoch number, will only run for one
    # but we save the given epoch for stats
    if one_epoch_mode:
        ori_epochs = num_epochs
        num_epochs = 1

    for epoch in range(num_epochs):

        epoch_loss, epoch_accuracy, epoch_f1 = 0.0, 0.0, 0.0
        loop_count = 0 # this track batch number (more robust than using batch_num)

        for data in data_loader:

            loop_count += 1

            # ----- create vars from inputs -----

            inputs = data[0] # (batch_num, instances, features)
            if len(inputs.shape)!=3:
                raise Exception('Only accept data with shape (B, N, F)')

            batch_num = inputs.shape[0]
            instance_num = inputs.shape[1]
            features_num = inputs.shape[-1]
            inputs = inputs.view(-1, features_num) # reshape -> (bag x instances, features)

            # ----- formatting labels -----

            if use_bag_labels and use_instance_labels==False:
                targets = data[1] # use bag level <- (batch_num)
                new_targets = []
                for target in targets: # <- make copy of labels for each bag
                    new_targets.append(target.repeat(instance_num))
                targets = torch.cat(new_targets)

            elif use_instance_labels and use_bag_labels==False:
                targets = data[2] # using instance label (batch_num, instances)
                targets = targets.flatten() # reshape -> (batch_num x instances)

            elif use_instance_labels and use_bag_labels:
                raise ValueError('Choose only one to be True; use_instance_labels or use_bag_labels')

            # check and convert target to long() dtype
            if targets.dtype != torch.long:
                targets = targets.long()

            # ----- forward/backward pass -----

            inputs, targets = inputs.to(device), targets.to(device)
            model.to(device)

            if validation_mode:
                model.eval()
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                model.train()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # ----- get predictions -----

            if device != 'cpu':
                outputs = outputs.cpu()
                targets = targets.cpu()

            outputs = outputs.reshape(batch_num, instance_num, -1) # reshape outputs back
            bag_predictions = torch.argmax(outputs.mean(dim=1), dim=1).numpy()  # Take mean over instances, then argmax over classes
            targets = targets[::instance_num].numpy() # unrepeat labels

            # ----- get metrics -----

            batch_accuracy = accuracy_score(targets, bag_predictions)
            batch_f1 = f1_score(targets, bag_predictions, average=f1_score_average)

            epoch_loss += loss.item()
            epoch_accuracy += batch_accuracy
            epoch_f1 += batch_f1

        # Calculate average metrics for the epoch
        epoch_loss /= loop_count # using <epoch_loss /= len(train_loader.dataset)> not good
        epoch_accuracy /= loop_count
        epoch_f1 /= loop_count

        # append all metrics to dictionary metrics
        if one_epoch_mode: # get back original epoch in this mode
            metrics['epoch'].append(ori_epochs)
            epoch = ori_epochs
        else:
            metrics['epoch'].append(epoch)
        metrics['loss'].append(epoch_loss)
        metrics['accuracy'].append(epoch_accuracy)
        metrics['f1'].append(epoch_f1)

        # ----- print and fname -----

        # print some outputs and create stats fname
        if validation_mode:
            text1 = 'VALIDATE'
            stat_fname = 'stats_val.json'
        else:
            text1 = 'TRAINING'
            stat_fname = 'stats_train.json'

        print(f"{text1} \t:| Epoch {epoch+1:03d} | Loss: {epoch_loss:.6f} | Accuracy: {epoch_accuracy:.4f} | F1: {epoch_f1:.4f} |")

        # condition to save metrics, save every epoch to be safe
        if savedir != None:
            dir_metrics = os.path.join(savedir, stat_fname)
            with open(dir_metrics, 'w') as json_file:
                json.dump(metrics, json_file, indent=4)

    return model, metrics
