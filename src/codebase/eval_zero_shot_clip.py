import glob
import hydra
import json
import logging
import os
import pickle
# from cxrclip.evaluator import Evaluator
from omegaconf import DictConfig, OmegaConf

from breastclip import seed_everything
from breastclip.evaluator import Evaluator

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="image_text_retrieval")
def main(cfg: DictConfig):
    seed_everything(cfg.base.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    OmegaConf.resolve(cfg)

    cfg_dict = OmegaConf.to_container(cfg)
    log.info(f"Configurations:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"Configurations in dict form:\n{cfg}")

    ckpt_path = cfg.model.clip_check_point
    save_path = cfg.base.output.save_path
    print(f"Checkpoint path: {ckpt_path}")
    print(f"Save path: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    pickle.dump(cfg, open(os.path.join(save_path, "args_dict.pkl"), "wb"))

    evaluator = Evaluator(cfg_dict, ckpt_path)
    print(cfg["data_test"])

    for test_dataset_name in cfg["data_test"]:
        log.info(f"Dataset: {test_dataset_name}")
        zs_prompts = cfg["base"]["zs_prompts"][test_dataset_name]
        print(f"Zero-shot prompts: {zs_prompts}")
        evals = {
            ckpt_path: evaluator.eval_zeroshot(
                ckpt_path, test_dataset_name, zs_prompts=zs_prompts, save_path=save_path)
        }

        with open(os.path.join(save_path, f"results-{test_dataset_name}.json"), "w") as outfile:
            json.dump(evals, outfile)
        log.info("print best score")


    log.info(f"All outputs are dumped in: {save_path}")


if __name__ == "__main__":
    main()
