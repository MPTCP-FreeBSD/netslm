"""
This file is rewritten based on openprompt.plms.__init__.py with small modifications
on model class initialization.
We write this file just to avoid direct coding on openprompt source codes.
"""

import math
from typing import List, Optional
from collections import namedtuple
from yacs.config import CfgNode
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import (
    BertConfig, BertTokenizer, BertLMHeadModel,
    RobertaConfig, RobertaTokenizer, RobertaForCausalLM,
    AlbertTokenizer, AlbertConfig, AlbertForMaskedLM,
    T5Config, T5Tokenizer, T5ForConditionalGeneration,
    OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, OpenAIGPTConfig,
    GPT2Config, GPT2Tokenizer,
    OPTConfig,
    ElectraConfig, ElectraForMaskedLM, ElectraTokenizer,
    GPTJConfig, GPTJForCausalLM,
    LlamaConfig, LlamaTokenizer, LlamaTokenizerFast,
    MistralConfig,
    AutoConfig, AutoTokenizer, AutoModelForCausalLM
)

from plm_special.models.gpt2 import GPT2Model
from plm_special.models.llama import LlamaModel
from plm_special.models.mistral import MistralModel
from plm_special.models.opt import OPTModel
from plm_special.models.t5 import T5Model

ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model'))

_MODEL_CLASSES = {
    'bert': ModelClass(**{
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model': BertLMHeadModel,
    }),
    'roberta': ModelClass(**{
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model': RobertaForCausalLM
    }),
    'albert': ModelClass(**{
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model': AlbertForMaskedLM,
    }),
    'gpt': ModelClass(**{
        'config': OpenAIGPTConfig,
        'tokenizer': OpenAIGPTTokenizer,
        'model': OpenAIGPTLMHeadModel
    }),
    'gpt2': ModelClass(**{
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer,
        'model': GPT2Model,
    }),
    't5': ModelClass(**{
        'config': T5Config,
        'tokenizer': T5Tokenizer,
        'model': T5ForConditionalGeneration,
    }),
    't5-lm': ModelClass(**{
        'config': T5Config,
        'tokenizer': T5Tokenizer,
        'model': T5Model,
    }),
    'opt': ModelClass(**{
        'config': OPTConfig,
        'tokenizer': GPT2Tokenizer,
        'model': OPTModel,
    }),
    'electra': ModelClass(**{
        'config': ElectraConfig,
        'tokenizer': ElectraTokenizer,
        'model': ElectraForMaskedLM,
    }),
    "gptj": ModelClass(**{
        "config": GPTJConfig, 
        "tokenizer": GPT2Tokenizer, 
        "model": GPTJForCausalLM,
    }),
    "llama": ModelClass(**{
        "config": LlamaConfig,
        "tokenizer": LlamaTokenizer,
        "model": LlamaModel,
    }),
    "mistral": ModelClass(**{
        "config": MistralConfig,
        "tokenizer": LlamaTokenizerFast,
        "model": MistralModel,
    }),
    "phi3": ModelClass(**{
        "config": AutoConfig,
        "tokenizer": AutoTokenizer,
        "model": AutoModelForCausalLM,
    }),
}


def get_model_class(plm_type: str):
    return _MODEL_CLASSES[plm_type]


def create_device_map_for_llama(device_input_side: str, device_output_side: str, device_middle_side: str = None):
    """
    Create device map for llama. The device map is used to evenly split the llama model into two/three parts on two devices.
    """
    device_map = {'embed_tokens': device_input_side}
    device_list = [device_input_side, device_output_side] if device_middle_side is None else [device_input_side, device_middle_side, device_output_side]
    for i in range(32):  # llama-7b has 32 transformer blocks
        device_map[f'layers.{i}'] = device_list[i // math.ceil(32 / len(device_list))]
    device_map['norm'] = device_output_side
    return device_map


def load_plm(model_name, model_path, specials_to_add=None, **kwargs):
    """
    Load a pre-trained language model (PLM), tokenizer, and configuration.

    Args:
        model_name (str): The type of the PLM (e.g., 'phi3').
        model_path (str): Path to the pre-trained model directory.
        specials_to_add (list, optional): Custom tokens to add. Defaults to None.

    Returns:
        tuple: model, tokenizer, and model configuration.
    """
    if 'phi3' in model_name:
        tokenizer_path = "/Users/raja/Documents/GitHub/netslm/downloaded_plms/Phi-3-mini-4k-instruct/tokenizers/microsoft/Phi-3-mini-4k-instruct"
        print(f"Loading Phi-3 tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        print(f"Loading Phi-3 model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    else:
        model_class = get_model_class(plm_type=model_name)
        model_config = model_class.config.from_pretrained(model_path)
        tokenizer = model_class.tokenizer.from_pretrained(model_path)
        model = model_class.model.from_pretrained(model_path, config=model_config)

    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=specials_to_add)

    return model, tokenizer, model.config


def add_special_tokens(model: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer,
                       specials_to_add: Optional[List[str]] = None):
    """
    Add special tokens to a tokenizer and resize the model embeddings.
    """
    if specials_to_add is None:
        return model, tokenizer
    for token in specials_to_add:
        if "pad" in token.lower() and tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': token})
            model.resize_token_embeddings(len(tokenizer))
            print("Pad token is None, set to ID {}".format(tokenizer.pad_token_id))
    return model, tokenizer
