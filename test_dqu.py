import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,6"
from utils import update_train_running_results, generate_randomized_fiq_caption, set_train_bar_description, save_model, device
import test
from statistics import mean, harmonic_mean, geometric_mean
from argparse import ArgumentParser
from models.ssn_crossAttention4 import Model

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
        "--mu", default=0.1, type=float
    )
    parser.add_argument(
        "--n_layers", default=4, type=int
    )
    parser.add_argument(
        "--n_heads", default=2, type=int
    )
    args = parser.parse_args()
    
    model = Model(args).to(device)
    state_dict = torch.load('/amax/home/chendian/SSN-master/SSN-master/src/outputs/cirr_dataset/ssn_crossAttention4_2024-09-12_20:37:02/saved_models/combiner_avg.pt', map_location=device)
    model_state_dict = {k: v for k, v in state_dict.items() if "epoch" not in k and "Model" not in k}
    model.load_state_dict(model_state_dict)
    
    model.eval()
    
    dataset_list_path = "/amax/home/chendian/DQU-CIR-main/dataset_list.pkl"
    # 检查文件是否存在
    if os.path.exists(dataset_list_path):
        # 读取 dataset_list
        with open(dataset_list_path, 'rb') as f:
            dataset_list = pickle.load(f)
    else:
        dataset_list = load_dataset()
        with open(dataset_list_path, 'wb') as f:
            pickle.dump(dataset_list, f)
    
    results = test.test_cirr_valset(args, model, dataset_list[0])
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
    print(results_dict)