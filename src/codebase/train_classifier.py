import warnings

import torch

from Classifiers.experiments import do_experiments
from utils import get_Paths, seed_all

warnings.filterwarnings("ignore")
import argparse
import os
import pickle


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard-path', metavar='DIR',
                        default='/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/log',
                        help='path to tensorboard logs')
    parser.add_argument('--checkpoints', metavar='DIR',
                        default='/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/checkpoints',
                        help='path to checkpoints')
    parser.add_argument('--output_path', metavar='DIR',
                        default='/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/out',
                        help='path to output logs')
    parser.add_argument(
        "--data-dir",
        default="/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset",
        type=str, help="Path to data file"
    )
    parser.add_argument(
        "--img-dir", default="RSNA_Cancer_Detection/train_images_png", type=str, help="Path to image file"
    )

    parser.add_argument("--clip_chk_pt_path", default=None, type=str, help="Path to Mammo-CLIP chkpt")
    parser.add_argument("--csv-file", default="RSNA_Cancer_Detection/final_rsna.csv", type=str,
                        help="data csv file")
    parser.add_argument("--dataset", default="RSNA", type=str, help="Dataset name? (RSNA or VinDr)")
    parser.add_argument("--data_frac", default=1.0, type=float, help="Fraction of data to be used for training")
    parser.add_argument(
        "--arch", default="upmc_breast_clip_det_b5_period_n_ft", type=str,
        help="For b5 classification, [upmc_breast_clip_det_b5_period_n_lp for linear probe and  upmc_breast_clip_det_b5_period_n_ft for finetuning]. "
             "For b2 classification, [upmc_breast_clip_det_b2_period_n_lp for linear probe and  upmc_breast_clip_det_b2_period_n_ft for finetuning].")
    parser.add_argument("--label", default="cancer", type=str,
                        help="cancer for RSNA or Mass, Suspicious_Calcification, density for VinDr")
    parser.add_argument("--detector-threshold", default=0.1, type=float)
    parser.add_argument("--swin_encoder", default="microsoft/swin-tiny-patch4-window7-224", type=str)
    parser.add_argument("--pretrained_swin_encoder", default="y", type=str)
    parser.add_argument("--swin_model_type", default="y", type=str)
    parser.add_argument("--VER", default="084", type=str)
    parser.add_argument("--epochs-warmup", default=0, type=float)
    parser.add_argument("--num_cycles", default=0.5, type=float)
    parser.add_argument("--alpha", default=10, type=float)
    parser.add_argument("--sigma", default=15, type=float)
    parser.add_argument("--p", default=1.0, type=float)
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555408335154, type=float)
    parser.add_argument("--focal-alpha", default=0.6, type=float)
    parser.add_argument("--focal-gamma", default=2.0, type=float)
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--n_folds", default=4, type=int)
    parser.add_argument("--start-fold", default=0, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--epochs", default=9, type=int)
    parser.add_argument("--lr", default=5.0e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--warmup-epochs", default=1, type=float)
    parser.add_argument("--img-size", nargs='+', default=[1520, 912])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--apex", default="y", type=str)
    parser.add_argument("--print-freq", default=5000, type=int)
    parser.add_argument("--log-freq", default=1000, type=int)
    parser.add_argument("--running-interactive", default='n', type=str)
    parser.add_argument("--inference-mode", default='n', type=str)
    parser.add_argument('--model-type', default="Classifier", type=str)
    parser.add_argument("--weighted-BCE", default='n', type=str)
    parser.add_argument("--balanced-dataloader", default='n', type=str)

    return parser.parse_args()


def main(args):
    seed_all(args.seed)
    # get paths
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # args.root = f"lr_{args.lr}_epochs_{args.epochs}_weighted_BCE_{args.weighted_BCE}_balanced_dataloader_{args.balanced_dataloader}_{args.label}_data_frac_{args.data_frac}_post_miccai"
    args.root = f"lr_{args.lr}_epochs_{args.epochs}_weighted_BCE_{args.weighted_BCE}_{args.label}_data_frac_{args.data_frac}"
    args.apex = True if args.apex == "y" else False
    args.pretrained_swin_encoder = True if args.pretrained_swin_encoder == "y" else False
    args.swin_model_type = True if args.swin_model_type == "y" else False
    args.running_interactive = True if args.running_interactive == "y" else False

    chk_pt_path, output_path, tb_logs_path = get_Paths(args)
    args.chk_pt_path = chk_pt_path
    args.output_path = output_path
    args.tb_logs_path = tb_logs_path

    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)
    print("====================> Paths <====================")
    print(f"checkpoint_path: {chk_pt_path}")
    print(f"output_path: {output_path}")
    print(f"tb_logs_path: {tb_logs_path}")
    print('device:', device)
    print('torch version:', torch.__version__)
    print("====================> Paths <====================")

    pickle.dump(args, open(os.path.join(output_path, f"seed_{args.seed}_train_configs.pkl"), "wb"))
    torch.cuda.empty_cache()

    if args.weighted_BCE == "y" and args.dataset.lower() == "rsna" and args.label.lower() == "cancer":
        args.BCE_weights = {
            "fold0": 46.48148148148148,
            "fold1": 46.01830663615561,
            "fold2": 46.41339491916859,
            "fold3": 46.05747126436781
        }
    elif args.weighted_BCE == "y" and args.dataset.lower() == "vindr" and args.label.lower() == "mass":
        args.BCE_weights = {
            "fold0": 15.573306370070778,
            "fold1": 15.573306370070778,
            "fold2": 15.573306370070778,
            "fold3": 15.573306370070778
        }
    elif args.weighted_BCE == "y" and args.dataset.lower() == "vindr" and args.label.lower() == "suspicious_calcification":
        args.BCE_weights = {
            "fold0": 37.296728971962615,
            "fold1": 37.296728971962615,
            "fold2": 37.296728971962615,
            "fold3": 37.296728971962615,
        }

    elif args.weighted_BCE == "y" and args.dataset.lower() == "vindr" and args.label.lower() == "focal_asymmetry":
        args.BCE_weights = {
            "fold0": 74.88425925925925,
            "fold1": 74.88425925925925,
            "fold2": 74.88425925925925,
            "fold3": 74.884259259259255,
        }

    if args.balanced_dataloader == "y":
        args.sampler_weights = {
            "fold0": {"pos_wt": 0.003401360544217687, "neg_wt": 7.469375560203167e-05},
            "fold1": {"pos_wt": 0.0035211267605633804, "neg_wt": 7.503001200480192e-05},
            "fold2": {"pos_wt": 0.003424657534246575, "neg_wt": 7.48839299086416e-05},
            "fold3": {"pos_wt": 0.003472222222222222, "neg_wt": 7.419498441905327e-05},
        }

    do_experiments(args, device)


if __name__ == "__main__":
    args = config()
    main(args)
