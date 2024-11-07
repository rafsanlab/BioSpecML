from sklearn.metrics import f1_score, accuracy_score
from ..utils.memory_utils import print_gpu_memory, print_cpu_memory
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import numpy as np

def train_mil_model(model, data_loader, device, num_epochs, criterion, optimizer=None,
                  scheduler=None,
                  micro_batch_patches:int=None,
                  savedir=None, f1_score_average='macro', labels=None, validation_mode=False,
                  use_instance_labels=True, use_bag_labels=False, 
                  pooling_method='mean', verbose=True, memory_verbose:bool=False,
                  one_epoch_mode=False, return_predictions=False):
    """
    Accept bag data (Bags, N-instances, Features) and do prediction based on 
    bag labels data either (Bags-labels, N-instances) or (Bags-labels).

    Args:
    - return_predictions(bool): return prediction by a single epoch, use with one_epoch_mode
    
    """
    
    model.to(device)
    if memory_verbose:
        print('Memory status after model send to device:')
        print_gpu_memory()
        print_cpu_memory()

    metrics = {'epochs':[], 'loss':[], 'accuracy':[],'f1':[]}

    # if this function is in another running loop, set one_epoch_mode to True
    # so regardless any epoch number, will only run for one
    # but we save the given epoch for stats
    if one_epoch_mode:
        ori_epochs = num_epochs
        num_epochs = 1

    for epoch in range(num_epochs):

        epoch_loss, epoch_accuracy, epoch_f1 = 0.0, 0.0, 0.0
        loop_count = 0 # this track batch number (more robust than using batch_num)
        epoch_bag_predictions = []

        for data in data_loader:

            loop_count += 1

            # ----- create vars from inputs -----

            inputs = data[0] # (batch_num, instances, features)
            # if len(inputs.shape)!=3:
            #     raise Exception('Only accept data with shape (B, N, F)')

            batch_num = inputs.shape[0]
            instance_num = inputs.shape[1]
            features_dim = tuple(inputs.shape[2:])
            inputs = inputs.view(batch_num*instance_num, *features_dim) # reshape -> (bag x instances, features)

            # ----- formatting labels -----

            if use_bag_labels and use_instance_labels==False:
                targets = data[1] # use bag level <- (batch_num)

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
            if memory_verbose:
                print('Memory status after data send to device:')
                print_gpu_memory()
                print_cpu_memory()

            if validation_mode:
                model.eval()
                with torch.no_grad():
                    if micro_batch_patches is not None:
                        for start in range(0, instance_num, micro_batch_patches):
                            micro_batch_data = inputs[:, start:start + micro_batch_patches, :, :, :]  # Shape: [2, 50, 3, 224, 224]
                            # Reshape to combine the batch and patch dimensions for model input
                            # Shape: [2 * 50, 3, 224, 224]
                            # micro_batch_data = micro_batch_data.view(-1, 3, 224, 224)
                            micro_batch_data = micro_batch_data.view(-1, *micro_batch_data.shape[1:])
                            outputs = model(micro_batch_data)
                            outputs = outputs.view(batch_num, -1, *outputs.shape[1:])
                            loss = criterion(outputs, targets)
                            loss.backward()
                    else:
                        outputs = model(inputs)
                        if use_bag_labels:
                            outputs = outputs.view(batch_num, instance_num, -1) # reconstruct data
                            if pooling_method=='mean':
                                outputs = torch.mean(outputs, dim=1) #-> (batch_num, -1)
                            elif pooling_method=='max':
                                outputs = torch.max(outputs, dim=1)[0]  # (batch_num, -1)
                        loss = criterion(outputs, targets)
            else:
                model.train()
                optimizer.zero_grad()
                if micro_batch_patches is not None:
                    for start in range(0, instance_num, micro_batch_patches):
                        micro_batch_data = inputs[:, start:start + micro_batch_patches, :, :, :]  # Shape: [2, 50, 3, 224, 224]
                        # Reshape to combine the batch and patch dimensions for model input
                        # Shape: [2 * 50, 3, 224, 224]
                        # micro_batch_data = micro_batch_data.view(-1, 3, 224, 224)
                        micro_batch_data = micro_batch_data.view(-1, *micro_batch_data.shape[1:])
                        outputs = model(micro_batch_data)
                        outputs = outputs.view(batch_num, -1, *outputs.shape[1:])
                        loss = criterion(outputs, targets)
                        loss.backward()
                else:
                    outputs = model(inputs)
                    if use_bag_labels:
                        outputs = outputs.view(batch_num, instance_num, -1) # reconstruct data
                        if pooling_method=='mean':
                            outputs = torch.mean(outputs, dim=1) #-> (batch_num, -1)
                        elif pooling_method=='max':
                            outputs = torch.max(outputs, dim=1)[0]  # (batch_num, -1)
                    loss = criterion(outputs, targets)
                    loss.backward()
                
                optimizer.step()
                if scheduler is not None:
                    scheduler.step(loss)

            # ----- get predictions -----

            if device != 'cpu':
                outputs = outputs.cpu()
                targets = targets.cpu()

            bag_predictions = torch.argmax(outputs, dim=1).numpy()
            if return_predictions:
                epoch_bag_predictions.append(bag_predictions)

            # ----- get metrics -----

            batch_accuracy = accuracy_score(targets, bag_predictions)
            batch_f1 = f1_score(targets, bag_predictions, average=f1_score_average, labels=labels, zero_division=np.nan)

            epoch_loss += loss.item()
            epoch_accuracy += batch_accuracy
            epoch_f1 += batch_f1

            # Set inputs and targets to None to help with memory
            torch.cuda.empty_cache()
            inputs, targets = None, None

        # Calculate average metrics for the epoch
        epoch_loss /= loop_count # using <epoch_loss /= len(train_loader.dataset)> not good
        epoch_accuracy /= loop_count
        epoch_f1 /= loop_count

        # append all metrics to dictionary metrics
        if one_epoch_mode: # get back original epoch in this mode
            metrics['epochs'].append(ori_epochs)
            epoch = ori_epochs
        else:
            metrics['epochs'].append(epoch)
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
        
        if verbose:
            print(f"{text1} \t:| Epoch {epoch+1:03d} | Loss: {epoch_loss:.6f} | Accuracy: {epoch_accuracy:.4f} | F1: {epoch_f1:.4f} |")

        # condition to save metrics, save every epoch to be safe
        if savedir != None:
            dir_metrics = os.path.join(savedir, stat_fname)
            with open(dir_metrics, 'w') as json_file:
                json.dump(metrics, json_file, indent=4)
    if return_predictions:
        return model, metrics, epoch_bag_predictions
    else:
        return model, metrics


def train_mil_val_loop(model, device, num_epochs, criterion, optimizer,
                    train_loader, test_loader=None, trained_num_epochs:int=None,
                    scheduler=None, micro_batch_patches=None,
                    verbose:bool=True, 
                    memory_verbose:bool=False,
                    f1_average:str='macro', labels=None, 
                    f1_average_test:str='macro', pooling_method='mean',
                    savedir:str=None, epoch_save_checkpoints:list=[],
                    save_model:bool=True,
                    ):
    """
    Example of use:
    >>> learning_rate , weight_decay = 0.001, 0
    >>> model, main_metrics = train_val_loop(
    >>>     model = model,
    >>>     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    >>>     num_epochs = 3,
    >>>     criterion = nn.CrossEntropyLoss(),
    >>>     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
    >>>     train_loader = train_loader,
    >>>     test_loader = test_loader,
    >>>     f1_average = 'macro',
    >>>     savedir = os.getcwd(),
    >>>     epoch_save_checkpoints = [2,3]
    >>> )
    """
    start_epoch = trained_num_epochs+1 if isinstance(trained_num_epochs, int) else 1
    main_metrics = {}
    main_metrics['epochs'] = []
    savedir = os.getcwd() if savedir is None or savedir == '' else savedir
    traindir = os.path.join(savedir, 'training')
    os.makedirs(traindir, exist_ok=True)

    for epoch in range(start_epoch, num_epochs+1, 1):

        # ----- run training and validation -----
        
        container_metrics = []
        
        if train_loader != None:
            model, train_metrics = train_mil_model(
                model = model,
                data_loader = train_loader,
                device = device,
                num_epochs = num_epochs,
                criterion = criterion,
                optimizer = optimizer,
                scheduler = scheduler,
                micro_batch_patches = micro_batch_patches,
                savedir = None,
                f1_score_average = f1_average,
                labels = labels,
                validation_mode = False,
                use_instance_labels = False,
                use_bag_labels = True,
                pooling_method = pooling_method,
                one_epoch_mode = True,
                verbose = False,
                memory_verbose = memory_verbose,
                )
            container_metrics.append(("train", train_metrics))

        if test_loader != None:
            model, test_metrics = train_mil_model(
                model = model,
                data_loader = test_loader,
                device = device,
                num_epochs = num_epochs,
                criterion = criterion,
                optimizer = optimizer,
                micro_batch_patches = micro_batch_patches,
                savedir = None,
                f1_score_average = f1_average_test,
                labels = labels,
                validation_mode = True,
                use_instance_labels = False,
                use_bag_labels = True,
                pooling_method = pooling_method,
                one_epoch_mode = True,
                verbose = False,
                memory_verbose = memory_verbose,
                )
            container_metrics.append(("val.", test_metrics))
        
        # ----- collect metrics -----

        main_metrics['epochs'].append(epoch)
        for phase, metrics in container_metrics:
            for k, v in metrics.items():
                new_k = f'{phase} {k}' # combine phase to dict's key
                if new_k not in main_metrics:
                    main_metrics[new_k] = v
                else:
                    main_metrics[new_k].extend(v)

        # print some metrics stats
        if verbose:
            print(f'Epoch {epoch:03d}', end=" : ")
            for phase, metrics in container_metrics:
                print(f'|| {phase.upper()}', end=' |')
                for key, value in metrics.items():
                    if 'epochs' not in key:
                        print(f"| {key} : {value[-1]:.6f}", end=" ")
            print()

        # save metrics
        stat_fname_path = os.path.join(traindir, f'stats_e{num_epochs}.json')
        with open(stat_fname_path, 'w') as json_file:
                json.dump(main_metrics, json_file, indent=4)

        # ----- option to save checkpoints -----

        if len(epoch_save_checkpoints)>0 and epoch in epoch_save_checkpoints:
            checkpoint_path = os.path.join(savedir, 'checkpoints')
            checkpoint_save_path = os.path.join(checkpoint_path, 'checkpoint_e'+str(epoch)+'.pth')
            checkpoint_stat_path = os.path.join(checkpoint_path, 'checkpoint_e'+str(epoch)+'_stats.json')
            if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path) 
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epochs': epoch,
                }
            torch.save(checkpoint, checkpoint_save_path)
            with open(checkpoint_stat_path, 'w') as json_file:
                json.dump(main_metrics, json_file, indent=4)
            print(f'Saved checkpoint at epoch {epoch}.')
 
    # ----- option to save model -----

    if save_model:
        model_path = os.path.join(traindir, f'model_e{epoch}.pth')
        torch.save(model, model_path)
        print(f'Saved model at epoch {epoch}.')

    return model, main_metrics
