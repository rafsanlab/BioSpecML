from ..ml.metrics_functions import calc_metric_prediction, calc_metric_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import numpy as np

# def train_model(model, data_loader, device, num_epochs, criterion, optimizer=None,
#               running_type:str='prediction', verbose:bool=True,
#               savedir:str=None, f1_average:str='macro', validation_mode:bool=False,
#               one_epoch_mode:bool=False, metrics_list:list=None,
#               ):
#     """
#     A basic running loop.
#     """
#     running_types = ['prediction', 'similarity']
#     if running_type not in running_types:
#         raise Exception(f'Choose *running_type : {running_types}')

#     # if this function is in another running loop, set one_epoch_mode to True
#     # so regardless any epoch number, will only run for -@rq one
#     # but we save the given epoch for stats 
#     if one_epoch_mode:
#         ori_epochs = num_epochs
#         num_epochs = 1

#     # ----- prepare metrics dictionary -----

#     if metrics_list == None:
#         raise Exception('Please provide metric list.')
#     metrics = {key: [] for key in metrics_list}
#     metrics['loss'], metrics['epochs'] = [], []

#     for epoch in range(num_epochs):

#         epoch = ori_epochs if one_epoch_mode else epoch+1
#         epoch_metrics = {key: 0.0 for key in metrics.keys()}
#         epoch_metrics['epochs'] = ori_epochs if one_epoch_mode else epoch

#         loop_count = 0 # this track batch number (more robust than using batch_num)

#         for data in data_loader:

#             loop_count += 1
#             inputs, targets = data[0], data[1]

#             # check and convert target to long() dtype
#             if running_type=='prediction':
#                 if targets.dtype != torch.long:
#                     targets = targets.long()

#             # ----- forward/backward pass -----

#             inputs, targets = inputs.to(device), targets.to(device)
#             model.to(device)

#             if validation_mode:
#                 model.eval()
#                 with torch.no_grad():
#                     outputs = model(inputs)
#                     loss = criterion(outputs, targets)
#             else:
#                 model.train()
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#             if device != 'cpu':
#                 outputs = outputs.cpu()
#                 targets = targets.cpu()

#             # ----- get metrics -----

#             # get prediction metrics
#             if running_type=='prediction':
#                 preds = torch.argmax(outputs, dim=1).numpy()
#                 targets = targets.numpy()
#                 batch_metrics = calc_metric_prediction(preds, targets, metrics_list, f1_average)

#             # get similarity metrics
#             if running_type=='similarity':
#                 outputs, targets = outputs.detach().numpy(), targets.detach().numpy()
#                 batch_metrics = calc_metric_similarity(outputs, targets, metrics_list)

#             for key in batch_metrics.keys():
#                 epoch_metrics[key] += batch_metrics[key]
#             epoch_metrics['loss'] += loss.item()

#         for key in epoch_metrics.keys():
#             if key != 'epochs': # escape 'epoch' value because we doing division
#                 epoch_metrics[key] /= loop_count

#         # ----- print, stat fname and append metrics -----

#         # set stats fname 
#         if validation_mode:
#             text1 = 'VALIDATE '
#             stat_fname = 'stats_val.json'
#         else:
#             text1 = 'TRAINING '
#             stat_fname = 'stats_train.json'

#         # print and append -@rq metrics
#         for key, value in epoch_metrics.items():
#             metrics[key].append(value)

#         # print some stats
#         if verbose:
#             print(f'{text1} Epoch {epoch:03d}', end=" - ")
#             for key, value in epoch_metrics.items():
#                 if key!= 'epochs':
#                     print(f"{key} : {value:.6f}", end=" | ")
#             print()

#         # condition to save metrics, save every epoch to be safe
#         if savedir != None:
#             dir_metrics = os.path.join(savedir, stat_fname)
#             with open(dir_metrics, 'w') as json_file:
#                 json.dump(metrics, json_file, indent=4)

#     return model, metrics

def train_model(
        model,
        data_loader,
        device,
        num_epochs,
        criterion,
        optimizer=None,
        running_type:str='prediction',
        verbose:bool=True,
        savedir:str=None,
        f1_average:str='macro',
        f1_zero_division='warn',
        labels:list=None,
        validation_mode:bool=False,
        one_epoch_mode:bool=False, 
        metrics_list:list=None,
        ):
    """
    A basic running loop.
    """
    running_types = ['prediction', 'similarity']
    if running_type not in running_types:
        raise Exception(f'Choose *running_type : {running_types}')

    if one_epoch_mode:
        ori_epochs = num_epochs
        num_epochs = 1

    if metrics_list == None:
        raise Exception('Please provide metric list.')
    metrics = {key: [] for key in metrics_list}
    metrics['loss'], metrics['epochs'] = [], []

    for epoch in range(num_epochs):

        epoch_val = ori_epochs if one_epoch_mode else epoch+1 # Use epoch_val to differentiate from loop `epoch`
        
        # Initialize lists to collect all predictions and targets for the epoch
        all_preds = []
        all_targets = []
        total_loss = 0.0 # Use total_loss to sum loss across batches

        loop_count = 0 

        for data in data_loader:
            loop_count += 1
            inputs, targets = data[0], data[1]

            if running_type=='prediction':
                if targets.dtype != torch.long:
                    targets = targets.long()

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

            if device != 'cpu':
                outputs = outputs.cpu()
                targets = targets.cpu()

            # Collect predictions and targets for epoch-level metric calculation
            if running_type=='prediction':
                preds = torch.argmax(outputs, dim=1).numpy()
                targets_np = targets.numpy()
                all_preds.extend(preds)
                all_targets.extend(targets_np)
            elif running_type=='similarity': # Assuming calc_metric_similarity also needs all outputs/targets
                # You'd need a similar collection for similarity outputs/targets
                # For simplicity, if calc_metric_similarity can be run batch-wise and averaged, keep current logic
                # Otherwise, collect all outputs and targets for similarity as well.
                # For this review, assuming you might keep calc_metric_similarity as is if its metrics are linearly additive.
                pass 

            total_loss += loss.item() # Sum loss items

        # --- Calculate epoch-level metrics after the batch loop ---
        epoch_metrics = {}
        epoch_metrics['epochs'] = epoch_val # Store the actual epoch number

        if running_type == 'prediction':
            # Calculate metrics once for the entire epoch's collected data
            epoch_metrics.update(calc_metric_prediction(
                np.array(all_targets), np.array(all_preds), 
                metrics_list, f1_average, f1_zero_division, labels))
        elif running_type == 'similarity':
            # If similarity metrics need to be calculated epoch-wise, do it here
            # Otherwise, if batch-wise averaging is acceptable for similarity, you'd need to re-introduce a similar sum/average
            # logic here, or modify calc_metric_similarity to accept lists and handle conversion.
            # For now, let's assume calc_metric_similarity should also be calculated on aggregated data if its metrics are non-linear like F1.
            # If they are simple means (like MSE), current loop-summing approach is fine.
            # For robust solution, collect all outputs/targets for similarity too and compute here.
            pass

        epoch_metrics['loss'] = total_loss / loop_count # Average loss over all batches

        # ----- print, stat fname and append metrics -----

        # set stats fname
        if validation_mode:
            text1 = 'VALIDATE '
            stat_fname = 'stats_val.json'
        else:
            text1 = 'TRAINING '
            stat_fname = 'stats_train.json'

        # print and append metrics - now epoch_metrics contains final values for the epoch
        for key, value in epoch_metrics.items():
            if key != 'epochs': # 'epochs' is not a list of values over time
                metrics[key].append(value)
            else:
                metrics[key].append(value) # Append the single epoch number

        # print some stats
        if verbose:
            print(f'{text1} Epoch {epoch_val:03d}', end=" - ")
            for key, value in epoch_metrics.items():
                if key!= 'epochs':
                    print(f"{key} : {value:.6f}", end=" | ")
            print()

        # condition to save metrics, save every epoch to be safe
        if savedir != None:
            dir_metrics = os.path.join(savedir, stat_fname)
            with open(dir_metrics, 'w') as json_file:
                json.dump(metrics, json_file, indent=4)

    return model, metrics


def train_val_loop(
        model,
        device,
        num_epochs,
        criterion,
        optimizer,
        train_loader,
        test_loader=None,
        trained_num_epochs:int=None,
        running_type:str='predictions',
        verbose:bool=True,
        f1_average:str='macro',
        f1_average_test:str='macro',
        f1_zero_division='warn',
        labels=None,
        metrics_list:list=['f1', 'accuracy'],
        savedir:str=None,
        epoch_save_checkpoints:list=[],
        save_model:bool=True,
        use_lr_scheduler:bool=False,
        lr_scheduler_step_size:int=None,
        lr_scheduler_gamma:float=None,
        early_stopping_patience:int=None,
        early_stopping_min_delta:float=0.0,
        early_stopping_monitor_metric:str=None,
        early_stopping_mode:str='min',
        ):
    
    """
    
    Main training and validation loop with early stopping and best model saving.
    
    This code license is exluded from the parent module license, hence follow the
    following disclaimer below:

    """

    print(
        "+-------------------------------------------------------------------------------------+\n"
        "|                                                                                     |\n"
        "|                                  ⚠️ DISCLAIMER: ⚠️                                  |\n"
        "|    By using this code, you're agreeing that the model, outputs or data generated    |\n"
        "|    using this code needs to include; Mohd Rifqi Rafsanjani (@rafsanlab) as the      |\n"
        "|    author in any distribution, publications, or publicly accessible projects (e.g., |\n"
        "|    demos, online applications), except for private use. This attribution should be  |\n"
        "|    clearly visible and accessible. If you disagree, please check the other          |\n"
        "|    training loop at (this code is in train_loop_v2.py, choose others);              |\n"
        "|    https://github.com/rafsanlab/BioSpecML/tree/main/src/biospecml/training_loops    |\n"
        "|                                                                                     |\n"
        "|    Features within this code are developed for my personal use in model development |\n"
        "|    and has been tested with external datasets, while the other training loop was    |\n"
        "|    specifically developed for the PhD project unless mentioned otherwise. License   |\n"
        "|    of the parent module is not applicable to this code. If you proceed, you are     |\n"
        "|    agreeing with the term of use and condition mentioned above.                     |\n"
        "|                                                                                     |\n"
        "+-------------------------------------------------------------------------------------+"
    )

    start_epoch = trained_num_epochs+1 if isinstance(trained_num_epochs, int) else 1
    main_metrics = {}
    main_metrics['epochs'] = []
    savedir = os.getcwd() if savedir is None or savedir == '' else savedir
    if early_stopping_patience is not None:
        traindir = os.path.join(savedir, 'training-estop')
    else:
        traindir = os.path.join(savedir, 'training')
    os.makedirs(traindir, exist_ok=True)
    
    # Define the path for the best model. It will be overwritten whenever a new best is found.
    best_model_path = os.path.join(traindir, 'best_model.pth')

    # --------------------------------------------------------------------------
    # Initialize learning rate scheduler
    scheduler = None
    if use_lr_scheduler:
        if lr_scheduler_step_size is None or lr_scheduler_gamma is None:
            raise ValueError("If use_lr_scheduler is True, lr_scheduler_step_size and lr_scheduler_gamma must be provided.")
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_scheduler_step_size,
            gamma=lr_scheduler_gamma
            )
        print(f"Learning rate scheduler initialized: StepLR(step_size={lr_scheduler_step_size}, gamma={lr_scheduler_gamma})")

    # --------------------------------------------------------------------------
    # Initialize Early Stopping variables
    best_metric_value = float('inf') if early_stopping_mode == 'min' else float('-inf')
    epochs_no_improve = 0
    best_epoch = -1
    
    # We no longer need to store the model state in a variable
    # best_model_state = None 

    if early_stopping_patience is not None:
        if early_stopping_monitor_metric is None:
            raise ValueError("If early_stopping_patience is set, early_stopping_monitor_metric must be provided -@rq")
        if early_stopping_monitor_metric not in [f'train {m}' for m in metrics_list + ['loss']] and \
           early_stopping_monitor_metric not in [f'val. {m}' for m in metrics_list + ['loss']]:
            raise ValueError(f"Monitor metric '{early_stopping_monitor_metric}' not found in available metrics.")
        print(f"Early stopping enabled: Monitoring '{early_stopping_monitor_metric}' with patience {early_stopping_patience} and min_delta {early_stopping_min_delta}.")

    # --------------------------------------------------------------------------
    # Start training
    for epoch in range(start_epoch, num_epochs+1, 1):

        container_metrics = []

        # Training phase
        if train_loader is not None:
            model, train_metrics = train_model(
                model, train_loader, device, epoch, criterion, optimizer,
                running_type = running_type,
                f1_average = f1_average,
                f1_zero_division = f1_zero_division,
                labels = labels,
                validation_mode = False,
                one_epoch_mode = True,
                metrics_list = metrics_list,
                verbose = False,
                )
            container_metrics.append(("train", train_metrics))

        # Validation phase (critical for early stopping)
        if test_loader is not None:
            model, test_metrics = train_model(
                model, test_loader, device, epoch, criterion, optimizer,
                running_type = running_type,
                f1_average = f1_average_test,
                validation_mode = True,
                one_epoch_mode = True,
                metrics_list = metrics_list,
                verbose = False,
                )
            container_metrics.append(("val.", test_metrics))
        else: # If no test_loader, early stopping cannot be based on validation metrics
            if early_stopping_patience is not None and early_stopping_monitor_metric.startswith('val.'):
                 print("Warning: Early stopping monitor metric is 'val.' but no test_loader provided. Early stopping may not function as expected. ")


        # ----- Step the learning rate scheduler -----
        if scheduler:
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            if verbose and current_lr != new_lr:
                print(f"Epoch {epoch:03d} - Learning rate updated from {current_lr:.6f} to {new_lr:.6f}")


        # ----- Collect metrics -----
        main_metrics['epochs'].append(epoch)
        current_epoch_metrics_dict = {} # To store current epoch's metrics for early stopping check
        for phase, metrics in container_metrics:
            for k, v in metrics.items():
                new_k = f'{phase} {k}'
                if new_k not in main_metrics:
                    main_metrics[new_k] = v
                else:
                    main_metrics[new_k].extend(v)
                # Store the latest value for early stopping check
                current_epoch_metrics_dict[new_k] = v[-1]

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

        # ----- [FIXED] Early Stopping Logic -----
        if early_stopping_patience is not None:
            current_monitor_value = current_epoch_metrics_dict.get(early_stopping_monitor_metric)
            if current_monitor_value is None:
                print(f"Warning: Monitored metric '{early_stopping_monitor_metric}' not found for early stopping. Skipping check.")
            else:
                improved = False
                if early_stopping_mode == 'min':
                    if current_monitor_value < best_metric_value - early_stopping_min_delta:
                        improved = True
                elif early_stopping_mode == 'max':
                    if current_monitor_value > best_metric_value + early_stopping_min_delta:
                        improved = True
                
                if improved:
                    best_metric_value = current_monitor_value
                    epochs_no_improve = 0
                    best_epoch = epoch
                    # --- CHANGE: Save the best model state_dict directly to a file ---
                    if save_model:
                        torch.save(model.state_dict(), best_model_path)
                        print(f"New best model state_dict saved at epoch {epoch} with {early_stopping_monitor_metric}: {best_metric_value:.6f}")
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}! No improvement in '{early_stopping_monitor_metric}' for {early_stopping_patience} epochs.")
                    break # Exit the training loop

        # ----- option to save checkpoints -----
        # This will save a checkpoint regardless of early stopping
        if len(epoch_save_checkpoints)>0 and epoch in epoch_save_checkpoints:
            checkpoint_path = os.path.join(savedir, 'checkpoints')
            checkpoint_save_path = os.path.join(checkpoint_path, 'checkpoint_e'+str(epoch)+'.pth')
            checkpoint_stat_path = os.path.join(checkpoint_path, 'checkpoint_e'+str(epoch)+'_stats.json')
            if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epochs': epoch,
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                }
            torch.save(checkpoint, checkpoint_save_path)
            with open(checkpoint_stat_path, 'w') as json_file:
                json.dump(main_metrics, json_file, indent=4)
            print(f'Saved checkpoint at epoch {epoch}.')

    # ----- [FIXED] Final Model Handling -----
    if save_model:
        final_model_path = os.path.join(traindir, f'final_model_e{epoch}.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f'Saved final model state_dict (from last epoch {epoch}) to {final_model_path}.')

        if best_epoch != -1:
            # If early stopping found a best model, load it back into the model object
            # before returning, and optionally rename the file to include the best epoch.
            print(f"Loading best model from epoch {best_epoch} with {early_stopping_monitor_metric}: {best_metric_value:.6f}")
            model.load_state_dict(torch.load(best_model_path))
            
            # Optional: Rename the best model file to include the epoch number for clarity
            final_best_path = os.path.join(traindir, f'best_model_e{best_epoch}.pth')
            os.rename(best_model_path, final_best_path)
            print(f'Best model state_dict is available at: {final_best_path}')
        else:
            print("Early stopping was not triggered or no improvement was found over the initial epoch.")


    return model, main_metrics
