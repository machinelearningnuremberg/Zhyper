import sys
from argparse import ArgumentParser
from hyper_llm_modulator.utils.eval_hypermod import eval_hypermod_checkpoint

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--full_eval", action="store_true")
    parser.add_argument("--use-icl", action="store_true")
    parser.add_argument("--new-eval", action="store_true")
    eval_hypermod_checkpoint(**vars(parser.parse_args()), curstep=None)
