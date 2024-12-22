import os
import sys
import numpy as np
import torch
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pprint import pprint
from munch import Munch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from config import cfg
from baseline_special.utils.utils import load_traces
from baseline_special.utils.constants import BITRATE_LEVELS
from plm_special.trainer import Trainer
from plm_special.evaluate import evaluate_on_env
from plm_special.test import test_on_env
from plm_special.data.dataset import ExperienceDataset
from plm_special.models.rl_policy import OfflineRLPolicy
from plm_special.models.state_encoder import EncoderNetwork
from plm_special.models.low_rank import peft_model
from plm_special.utils.utils import set_random_seed
from plm_special.utils.plm_utils import load_plm
from plm_special.utils.console_logger import ConsoleLogger

# Ensure paths are correct for the Phi-3 model
PLM_PATH = "/Users/raja/Documents/GitHub/netslm/downloaded_plms/phi3/base"
TOKENIZER_PATH = "/Users/raja/Documents/GitHub/netslm/downloaded_plms/Phi-3-mini-4k-instruct/tokenizers/microsoft/Phi-3-mini-4k-instruct"

PLM_LAYER_SIZES = {
    'gpt2': {'base': 24, 'small': 12, 'large': 36, 'xl': 48},
    'llama': {'base': 32},
    't5-lm': {'base': 12, 'small': 6, 'large': 24, 'xl': 24},
    'phi3': {'base': 32},  # Layer size for Phi-3
}

def save_model(args, model, save_dir):
    if args.rank > 0:
        model.plm.save_pretrained(save_dir)
        torch.save(model.modules_except_plm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))

def load_model(args, model, model_dir):
    if args.rank > 0:
        model.plm.load_adapter(model_dir, adapter_name='default')
        model.modules_except_plm.load_state_dict(torch.load(os.path.join(model_dir, 'modules_except_plm.bin')))
    else:
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))
    return model

def adapt(args, model, exp_dataset, exp_dataset_info, eval_env_settings, checkpoint_dir, best_model_dir, eval_process_reward_fn):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = LambdaLR(optimizer, lambda steps: min((steps + 1) / args.warmup_steps, 1))
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(args, model=model, optimizer=optimizer, exp_dataset=exp_dataset, loss_fn=loss_fn, 
                      device=args.device, lr_scheduler=lr_scheduler, grad_accum_steps=args.grad_accum_steps)

    target_return = exp_dataset_info.max_return * args.target_return_scale
    best_eval_return = 0.0

    for epoch in range(args.num_epochs):
        train_logs, train_losses = trainer.train_epoch()
        print(f"Training Iteration #{epoch}")
        pprint(train_logs)

        if epoch % args.eval_per_epoch == 0:
            eval_logs = evaluate_on_env(args, env_settings=eval_env_settings, model=model, target_return=target_return, 
                                        max_ep_num=args.trace_num, process_reward_fn=eval_process_reward_fn)
            pprint(eval_logs)
            if eval_logs['episodes_return'] > best_eval_return:
                best_eval_return = eval_logs['episodes_return']
                save_model(args, model, best_model_dir)

def run(args):
    assert args.plm_type in cfg.plm_types
    assert args.plm_size in cfg.plm_sizes

    set_random_seed(args.seed)

    trace_dir = cfg.trace_dirs[args.trace]
    video_size_dir = cfg.video_size_dirs[args.video]
    all_cooked_time, all_cooked_bw, all_file_names, all_mahimahi_ptrs = load_traces(trace_dir)

    exp_pool = pickle.load(open(args.exp_pool_path, 'rb'))
    exp_dataset = ExperienceDataset(exp_pool, gamma=args.gamma, scale=args.scale, max_length=args.w, sample_step=args.sample_step)
    exp_dataset_info = Munch(exp_dataset.exp_dataset_info)

    print(f"Loading PLM model from: {PLM_PATH}")
    plm, *_ = load_plm(args.plm_type, PLM_PATH, device_input_side=args.device, tokenizer_path=TOKENIZER_PATH)
    if args.plm_type != 'llama':
        plm = plm.to(args.device)

    if args.adapt:
        checkpoint_dir = f"{args.plm_type}_{args.plm_size}_checkpoint"
        best_model_dir = f"{args.plm_type}_{args.plm_size}_best_model"
        adapt(args, plm, exp_dataset, exp_dataset_info, {}, checkpoint_dir, best_model_dir, lambda x: x)

if __name__ == '__main__':
    parser = ArgumentParser(description="Run PLM script", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--adapt', action="store_true", help="Fine-tune model")
    parser.add_argument('--test', action="store_true", help="Test the model")
    parser.add_argument('--plm-type', type=str, default='phi3', help="Type of PLM (e.g., phi3, llama)")
    parser.add_argument('--plm-size', type=str, default='base', help="Size of PLM")
    parser.add_argument('--device', type=str, default='cpu', help="Device to run on")
    parser.add_argument('--exp-pool-path', type=str, required=True, help="Path to the experience pool file")
    parser.add_argument('--grad-accum-steps', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('--rank', type=int, default=-1, help="Rank of low-rank matrices")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--warmup-steps', type=int, default=2000, help="Warmup steps")
    parser.add_argument('--num-epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--eval-per-epoch', type=int, default=1, help="Evaluate per number of epochs")
    parser.add_argument('--trace', type=str, required=True, help="Trace name (e.g., fcc-train)")
    parser.add_argument('--video', type=str, required=True, help="Video name (e.g., video1)")
    parser.add_argument('--gamma', type=float, default=1.0, help="Discount factor for rewards")
    parser.add_argument('--scale', type=int, default=1000, help="Scale for rewards/returns")
    parser.add_argument('--w', type=int, default=20, help="Context window for learning return distribution")
    parser.add_argument('--sample-step', type=int, default=1, help="Sampling steps for experience")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")

    args = parser.parse_args()
    print("Arguments:")
    pprint(vars(args))
    run(args)
