from comet_ml import Experiment
import json
import multiprocessing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,5"
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, harmonic_mean, geometric_mean
from typing import List
import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import models
import utils

from data_utils import out_path, CIRRDataset
from utils import collate_fn, update_train_running_results, set_train_bar_description, save_model, device, ShowBestCIRR
from validate import compute_cirr_val_metrics
def combiner_training_cirr(args):
    """
    Train the Combiner on CIRR dataset
    """
    # Make training folder
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        out_path / 'outputs' / f"{args.dataset}" / f"{args.model}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)
    
    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    model = models.setup(args).to(device, non_blocking=True)

    def preprocess(im):
        return model.processor(images=im, return_tensors='pt')

    # Define the validation datasets and extract the validation index features
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    relative_train_dataset = CIRRDataset('train', 'relative', preprocess)
    # results = compute_cirr_val_metrics(classic_val_dataset, relative_val_dataset, model)

    # Define the model and the train dataset
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=args.batch_size,
                                       num_workers=4,
                                       pin_memory=True, collate_fn=collate_fn, drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    clip_params, proj_params, modality_params, fusion_params, head_params, co_params, sa_params = [], [], [], [], [], [], []
    for name, para in model.named_parameters():
        if 'clip' in name:
            clip_params += [para]
        elif 'token_proj' in name:
            proj_params += [para]
        elif 'token_type' in name:
            modality_params += [para]
        elif 'fusion' in name or 'token_selection' in name:
            fusion_params += [para]
        elif 'crossAttention' in name:
            co_params += [para]
        elif 'sa' in name:
            sa_params += [para]
        else:
            head_params += [para]
    optimizer = optim.Adam([{'params': head_params},
                            {'params': fusion_params, 'lr': args.lr * args.lr_ratio * 4},
                            {'params': modality_params, 'lr': args.lr * args.lr_ratio * 4},
                            {'params': proj_params, 'lr': args.lr * args.lr_ratio * 2},
                            {'params': clip_params, 'lr': args.lr * args.lr_ratio},
                            {'params': co_params, 'lr': args.lr_co},
                            {'params': sa_params, 'lr': args.lr_sa},
                            ], lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    scaler = torch.cuda.amp.GradScaler()
    
    show_best = ShowBestCIRR()

    # When save_best == True initialize the best results to zero
    if args.save_best:
        best_harmonic = 0
        best_geometric = 0
        best_arithmetic = 0
    best_avg_recall = 0
        
    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(args.num_epochs):
        with experiment.train():
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            model.train()
            train_bar = tqdm(relative_train_loader, ncols=150)
            for idx, (reference_images, target_images, texts) in enumerate(train_bar):  # Load a batch of triplets
                step = len(train_bar) * epoch + idx
                num_samples = len(texts)

                optimizer.zero_grad()
                show_best(epoch, best_avg_recall, best_harmonic, best_geometric, best_arithmetic)

                reference_images = reference_images.to(device)
                target_images = target_images.to(device)
                text_inputs = model.tokenizer(texts, padding=True, return_tensors='pt').to(device)

                # Compute the logits and loss
                with torch.cuda.amp.autocast():
                    ground_truth = torch.arange(num_samples, dtype=torch.long, device=device)
                    loss = model(reference_images, text_inputs, target_images, ground_truth)

                # Backpropagate and update the weights
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, num_samples)
                set_train_bar_description(train_bar, epoch, args.num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            scheduler.step()
            experiment.log_metric('learning_rate', scheduler.get_last_lr()[0], epoch=epoch)

        if epoch % args.validation_frequency == 0:
            with experiment.validate():
                model.eval()

                # Compute and log validation metrics
                results = compute_cirr_val_metrics(classic_val_dataset, relative_val_dataset, model)
                group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results

                results_dict = {
                    'group_recall_at1': group_recall_at1,
                    'group_recall_at2': group_recall_at2,
                    'group_recall_at3': group_recall_at3,
                    'recall_at1': recall_at1,
                    'recall_at5': recall_at5,
                    'recall_at10': recall_at10,
                    'recall_at50': recall_at50,
                    'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
                    'arithmetic_mean': mean(results),
                    'harmonic_mean': harmonic_mean(results),
                    'geometric_mean': geometric_mean(results)
                }

                print(json.dumps(results_dict, indent=4))
                
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)
                
                experiment.log_metrics(
                    results_dict,
                    epoch=epoch
                )

                # Save model
                if args.save_training:
                    if args.save_best and results_dict['arithmetic_mean'] > best_arithmetic:
                        best_arithmetic = results_dict['arithmetic_mean']
                        # save_model('combiner_arithmetic', epoch, model, training_path)
                    if args.save_best and results_dict['harmonic_mean'] > best_harmonic:
                        best_harmonic = results_dict['harmonic_mean']
                        # save_model('combiner_harmonic', epoch, model, training_path)
                    if args.save_best and results_dict['geometric_mean'] > best_geometric:
                        best_geometric = results_dict['geometric_mean']
                        # save_model('combiner_geometric', epoch, model, training_path)
                    if args.save_best and results_dict['mean(R@5+R_s@1)'] > best_avg_recall:
                        best_json_path = os.path.join(training_path, "metrics_best.json")
                        utils.save_dict_to_json(results_dict, best_json_path)
                        best_avg_recall = results_dict['mean(R@5+R_s@1)']
                        save_model('combiner_avg', epoch, model, training_path)
                    if not args.save_best:
                        save_model(f'combiner_{epoch}', epoch, model, training_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--model", type=str, required=True, help="could be 'FusionModel'")
    parser.add_argument("--clip_model_name", default="openai/clip-vit-base-patch32", type=str, help="CLIP model to use")

    parser.add_argument("--projection_dim", default=1024, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden_dim", default=2048, type=int, help="Combiner hidden dim")

    parser.add_argument("--lr", default=4e-5, type=float, help="Init learning rate")
    parser.add_argument("--lr_co", default=4e-5, type=float, help="Init learning rate")
    parser.add_argument("--lr_sa", default=4e-5, type=float, help="Init learning rate")
    parser.add_argument("--lr_exter", default=4e-5, type=float, help="Init learning rate")
    parser.add_argument("--lr_step_size", default=5, type=int)
    parser.add_argument("--lr_gamma", default=0.1, type=float)
    parser.add_argument("--lr_ratio", default=0.2, type=float)

    parser.add_argument("--num_epochs", default=100, type=int, help="number training epochs")
    parser.add_argument("--batch_size", default=384, type=int, help="Batch size of the Combiner training")
    parser.add_argument("--validation_frequency", default=1, type=int, help="Validation frequency expressed in epochs")

    parser.add_argument("--save_training", action='store_true', help="Whether save the training model")
    parser.add_argument("--save_best", action='store_true', help="Save only the best model during training")

    # comet.ml
    parser.add_argument("--api_key", type=str, default='xxx', help="api for Comet logging")
    parser.add_argument("--workspace", type=str, help="workspace of Comet logging")
    parser.add_argument("--project_name", default="xxx", type=str, help="name of the project on Comet")
    parser.add_argument("--kl_weight", default=1, type=int)
    parser.add_argument(
        "--embed-size", default=512, type=int
    )
    parser.add_argument(
        "--n_layers", default=4, type=int
    )
    parser.add_argument(
        "--n_heads", default=2, type=int
    )
    parser.add_argument(
        "--mu", default=0.1, type=float
    )

    args = parser.parse_args()
    
    training_hyper_params = {
    "dataset": args.dataset,
    "model": args.model,
    "clip_model_name": args.clip_model_name,
    "projection_dim": args.projection_dim,
    "hidden_dim": args.hidden_dim,
    "lr": args.lr,
    "lr_co": args.lr_co,
    "lr_sa": args.lr_sa,
    "lr_exter": args.lr_exter,
    "lr_step_size": args.lr_step_size,
    "lr_gamma": args.lr_gamma,
    "lr_ratio": args.lr_ratio,
    "num_epochs": args.num_epochs,
    "batch_size": args.batch_size,
    "validation_frequency": args.validation_frequency,
    "save_training": args.save_training,
    "save_best": args.save_best,
    "kl_weight": args.kl_weight,
    "embed_size": args.embed_size,
    "mu": args.mu,
    "n_layers": args.n_layers,
    "n_heads": args.n_heads
}

    if args.api_key and args.workspace:
        print("Comet logging ENABLED")
        experiment = Experiment(
            api_key=args.api_key,
            project_name=args.project_name,
            workspace=args.workspace,
            disabled=False
        )
    else:
        print("Comet loging DISABLED, to enable it provide an api key and a workspace")
        experiment = Experiment(
            api_key="",
            project_name="",
            workspace="",
            disabled=True
        )

    experiment.log_code(folder=str(out_path / 'code'))
    experiment.log_parameters(args)

    combiner_training_cirr(args)

