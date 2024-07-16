import warnings

from Detectors.experiments import do_experiements
from utils import get_Paths, seed_all

warnings.filterwarnings("ignore")
import argparse
import os
import torch
import torchvision
import pickle


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard-path', metavar='DIR',
                        default='/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/log',
                        help='path to tensorboard logs')
    parser.add_argument('--checkpoints', metavar='DIR',
                        default='/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/checkpoints',
                        help='path to checkpoints')
    parser.add_argument('--output_path', metavar='DIR',
                        default='/restricted/projectnb/batmanlab/shawn24/PhD/Multimodal-mistakes-debug/out',
                        help='path to output logs')
    parser.add_argument(
        "--data-dir",
        default="/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset",
        type=str, help="Path to data file"
    )
    parser.add_argument(
        "--img-dir", default="External/UPMC/DICOM/images_png_CC_MLO", type=str, help="Path to image file"
    )
    parser.add_argument("--csv-file", default="External/UPMC/upmc_dicom_consolidated_folds.csv", type=str,
                        help="data csv file")
    parser.add_argument("--clip_chk_pt_path", default=None, type=str, help="Path to Mammo-CLIP chkpt")
    parser.add_argument("--dataset", default="ViNDr", type=str, help="Dataset name?")
    parser.add_argument("--data_frac", default=1.0, type=float, help="Fraction of data to be used for training")
    parser.add_argument(
        "--freeze_backbone", default='n', type=str,
        help="Freeze backbone? (y for yes, n for no). If freezes, only the head is only trained")
    parser.add_argument("--arch", default="clip_b5_upmc", type=str,
                        help="Model architecture, (clip_b5_upmc or clip_b2_upmc)")
    parser.add_argument("--iou-threshold", default=0.5, type=float)
    parser.add_argument("--score-threshold", default=0.05, type=float)
    parser.add_argument("--epochs-warmup", default=0, type=float)
    parser.add_argument("--max-detections", default=100, type=int)
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
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--epochs", default=7, type=int)
    parser.add_argument("--lr", default=5.0e-5, type=float)
    parser.add_argument("--detection-threshold", default=0.3, type=float)
    parser.add_argument("--img-size", nargs='+', default=[1520, 912])
    parser.add_argument("--resize", default=512, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument('--model-type', default="Concept-Detector", type=str)
    parser.add_argument("--apex", default="y", type=str)
    parser.add_argument("--print-freq", default=100, type=int)
    parser.add_argument("--log-freq", default=100, type=int)
    parser.add_argument("--running-interactive", default='n', type=str)
    parser.add_argument(
        "--concepts", nargs='+', default=[
            'No Finding',
            'Architectural Distortion',
            'Asymmetry',
            'Focal Asymmetry',
            'Global Asymmetry',
            'Mass',
            'Nipple Retraction',
            'Skin Retraction',
            'Skin Thickening',
            'Suspicious Calcification',
            'Suspicious Lymph Node',
        ], help="which label of VinDr to detect?"
    )

    return parser.parse_args()


def main(args):
    print("PyTorch Version:", torch.__version__)
    print("Torchvision Version:", torchvision.__version__)
    seed_all(args.seed)
    # get paths
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.root = f"lr_{args.lr}_epochs_{args.epochs}_concepts_{args.concepts[0]}_alpha_{args.focal_alpha}_gamma_{args.focal_gamma}_score_th_{args.score_threshold}_data_frac_{args.data_frac}_freeze_backbone_{args.freeze_backbone}"

    args.apex = True if args.apex == "y" else False
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
    print(f'No. of concept: {len(args.concepts)}')
    print("====================> Paths <====================")

    pickle.dump(args, open(os.path.join(output_path, f"seed_{args.seed}_train_configs.pkl"), "wb"))
    torch.cuda.empty_cache()

    do_experiements(args, device)


if __name__ == "__main__":
    args = config()
    main(args)
