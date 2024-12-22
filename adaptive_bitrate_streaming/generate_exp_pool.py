import argparse
import os
import pickle
import itertools
import random
import numpy as np
import tensorflow as tf
from numba import jit
from config import cfg
import baseline_special.a3c as a3c
import baseline_special.env as env
from baseline_special.utils.utils import load_traces
from baseline_special.utils.constants import (
    REBUF_PENALTY, SMOOTH_PENALTY, DEFAULT_QUALITY, S_INFO, S_LEN, A_DIM, BITRATE_LEVELS, BUFFER_NORM_FACTOR,
    M_IN_K, SMOOTH_PENALTY, VIDEO_BIT_RATE, CHUNK_TIL_VIDEO_END_CAP, RAND_RANGE, DEFAULT_QUALITY, TOTAL_VIDEO_CHUNK
)
from plm_special.utils.utils import action2bitrate
from plm_special.data.exp_pool import ExperiencePool
from plm_special.utils.plm_utils import load_plm  # Added for PLM integration

PENSIEVE = 0
MPC = 1
BBA = 2
PLM = 3  # New constant for PLM integration

# =========================================================================
# ====================== Pensieve Special (Start) =========================
# =========================================================================

def pensieve(actor, state, last_bit_rate):
    action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
    action_cumsum = np.cumsum(action_prob)
    action = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
    bit_rate = action2bitrate(action, last_bit_rate)
    return bit_rate

# =========================================================================
# ======================= Pensieve Special (End) ==========================
# =========================================================================

# =========================================================================
# ========================= MPC Special (Start) ===========================
# =========================================================================

CHUNK_COMBO_OPTIONS = np.array([combo for combo in itertools.product(range(6), repeat=5)])
MPC_FUTURE_CHUNK_COUNT = 5

@jit(nopython=True)
def next_possible_bitrates(br):
    next_brs = [br - 1, br, br + 1]
    next_brs = [a for a in next_brs if 0 <= a <= 5]
    return next_brs

# Additional JIT functions omitted for brevity...

def mpc(*args):
    # MPC logic omitted for brevity...
    pass

# =========================================================================
# ========================== MPC Special (End) ============================
# =========================================================================

# =========================================================================
# ========================= BBA Special (Start) ===========================
# =========================================================================

RESEVOIR = 5  # BBA
CUSHION = 10  # BBA

def bba(buffer_size):
    if buffer_size < RESEVOIR:
        bit_rate = 0
    elif buffer_size >= RESEVOIR + CUSHION:
        bit_rate = BITRATE_LEVELS - 1
    else:
        bit_rate = (BITRATE_LEVELS - 1) * (buffer_size - RESEVOIR) / float(CUSHION)
    bit_rate = int(bit_rate)
    return bit_rate

# =========================================================================
# ========================== BBA Special (End) ============================
# =========================================================================


def collect_experience(args, model, model_name, env_settings, trace_num, sess=None, actor=None, plm=None):
    net_env = env.Environment(**env_settings)
    total_states, total_actions, total_rewards, total_dones = [], [], [], []
    state = np.zeros((S_INFO, S_LEN), dtype=np.float32)
    bit_rate = DEFAULT_QUALITY
    last_bit_rate = DEFAULT_QUALITY

    while True:
        delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)

        # Update state and reward (omitted for brevity)

        if model == PENSIEVE:
            bit_rate = pensieve(actor, state, last_bit_rate)
        elif model == MPC:
            bit_rate = mpc(...)  # Parameters omitted for brevity
        elif model == PLM:
            inputs = plm['tokenizer'](state, return_tensors='pt').to(args.device)
            logits = plm['model'](**inputs).logits
            action = torch.argmax(logits, dim=-1).item()
            bit_rate = action2bitrate(action, last_bit_rate)
        else:
            bit_rate = bba(buffer_size)

        # Append to experience pool (omitted for brevity)

        if end_of_video:
            break

    return total_states, total_actions, total_rewards, total_dones


def run(args):
    assert args.trace in cfg.trace_dirs.keys()
    assert args.video in cfg.video_size_dirs.keys()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_id)  
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    exp_pools_dir = os.path.join(cfg.exp_pools_dir, args.trace + f'_{args.video}', '_'.join(args.models), f'seed_{args.seed}_trace_num_{args.trace_num}_fixed_{args.fixed_order}')
    os.makedirs(exp_pools_dir, exist_ok=True)
    exp_pool = ExperiencePool()

    plm = None
    if 'phi3' in args.models:
        plm_model, plm_tokenizer, _ = load_plm('phi3', os.path.join(cfg.plm_dir, 'phi3', 'base'))
        plm = {'model': plm_model, 'tokenizer': plm_tokenizer}

    for model_name in args.models:
        model = PENSIEVE if model_name in ['genet', 'udr_1', 'udr_2', 'udr_3', 'udr_real'] else MPC if model_name == 'mpc' else BBA if model_name == 'bba' else PLM
        states, actions, rewards, dones, _ = collect_experience(args, model, model_name, env_settings={}, trace_num=args.trace_num, plm=plm)
        for i in range(len(states)):
            exp_pool.add(state=states[i], action=actions[i], reward=rewards[i], done=dones[i])

    exp_pool_path = os.path.join(exp_pools_dir, 'exp_pool.pkl')
    pickle.dump(exp_pool, open(exp_pool_path, 'wb'))
    print(f"Done. Experience pool saved at:", exp_pool_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', help='choose one or more from [genet, udr_1, udr_2, udr_3, udr_real, mpc, bba, phi3]', nargs='*', default='genet')
    parser.add_argument("--trace", help='name of traces (e.g., fcc-train)')
    parser.add_argument('--video', help='name of videos (e.g., video1)')
    parser.add_argument('--trace-num', type=int, help='number of traces. if set to -1, use all traces in the trace dir.', default=-1)
    parser.add_argument('--seed', type=int, help='random seed', default=100003)
    parser.add_argument('--cuda-id', type=int, help='cuda device idx', default=0)
    parser.add_argument('--fixed-order', action='store_true', help='iterate over test traces in a fixed sequential order.')
    args = parser.parse_args()

    run(args)
