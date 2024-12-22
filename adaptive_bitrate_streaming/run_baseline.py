""" Tensorflow v1 is required to run this file. """
import argparse
import os
import itertools
import numpy as np
import tensorflow as tf
import random
from numba import jit
from config import cfg
from baseline_special.utils.utils import load_traces
from baseline_special.utils.constants import (
    REBUF_PENALTY, SMOOTH_PENALTY, DEFAULT_QUALITY, S_INFO, S_LEN, A_DIM, BITRATE_LEVELS, BUFFER_NORM_FACTOR,
    M_IN_K, VIDEO_BIT_RATE, CHUNK_TIL_VIDEO_END_CAP, TOTAL_VIDEO_CHUNK
)
from plm_special.utils.utils import action2bitrate, calc_mean_reward, clear_dir
from plm_special.utils.plm_utils import load_plm

PENSIEVE = 0
MPC = 1
BBA = 2

# Explicitly disable GPU for CPU-only environments
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# =========================================================================
# ====================== Phi-3 Integration (Start) ========================
# =========================================================================
def phi3_decision(state, tokenizer, model):
    """
    Use the Phi-3 model to decide the next bitrate based on the current state.
    """
    input_text = f"State: {state.tolist()}"
    inputs = tokenizer(input_text, return_tensors="pt")  # Ensure on CPU
    outputs = model.generate(inputs.input_ids, max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        # Parse bitrate from the Phi-3 response (assumes a numerical bitrate is returned)
        next_bitrate = int(response.strip().split()[-1])  # Adjust parsing logic as needed
    except ValueError:
        print(f"Error parsing response: {response}")
        next_bitrate = DEFAULT_QUALITY

    return next_bitrate

# =========================================================================
# ====================== Phi-3 Integration (End) ==========================
# =========================================================================


def run(args):
    assert args.model in ['genet', 'udr_1', 'udr_2', 'udr_3', 'udr_real', 'mpc', 'bba', 'phi3']
    print(cfg.trace_dirs.keys())
    print(args.test_trace)
    assert args.test_trace in cfg.trace_dirs.keys()
    assert args.video in cfg.video_size_dirs.keys()

    trace_dir = cfg.trace_dirs[args.test_trace]
    video_size_dir = cfg.video_size_dirs[args.video]

    all_cooked_time, all_cooked_bw, all_file_names, all_mahimahi_ptrs = load_traces(trace_dir)

    net_env = env.Environment(
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw,
        all_file_names=all_file_names,
        all_mahimahi_ptrs=all_mahimahi_ptrs,
        video_size_dir=video_size_dir,
        fixed=args.fixed_order,
        trace_num=min(args.test_trace_num, len(all_file_names))
    )

    if args.model == "phi3":
        print("Loading Phi-3 model and tokenizer...")
        model_path = "/Users/raja/Documents/GitHub/netslm/downloaded_plms/Phi-3-mini-4k-instruct"
        tokenizer, model, config = load_plm("phi3", model_path)
        print("Phi-3 model loaded successfully.")

    results_dir = os.path.join(
        cfg.results_dir, f'{args.test_trace}_{args.video}',
        f'trace_num_{args.test_trace_num}_fixed_{args.fixed_order}', args.model, f'seed_{args.seed}'
    )
    os.makedirs(results_dir, exist_ok=True)
    clear_dir(results_dir)

    time_stamp = 0
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    state = np.zeros((S_INFO, S_LEN), dtype=np.float32)

    with tf.Session() as sess:
        while True:
            delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)

            time_stamp += delay + sleep_time

            # Update state
            state = np.roll(state, -1, axis=1)
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
            state[4, :BITRATE_LEVELS] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
            state[5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            if args.model == "phi3":
                bit_rate = phi3_decision(state, tokenizer, model)
            elif args.model == "mpc":
                bit_rate = mpc(state, bit_rate, buffer_size)
            elif args.model == "bba":
                bit_rate = bba(buffer_size)

            if end_of_video:
                break

        print("Testing complete.")
