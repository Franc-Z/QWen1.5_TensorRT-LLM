# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import configparser
import time
from operator import attrgetter
from pathlib import Path

import numpy as np

from safetensors import safe_open
from tqdm import tqdm
#from transformers import AutoModelForCausalLM


import torch
#import torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix
import tensorrt_llm
from tensorrt_llm._utils import (str_dtype_to_np, str_dtype_to_torch,numpy_to_torch,torch_to_numpy)
from tensorrt_llm.mapping import Mapping
from model import QWenForCausalLM
from tensorrt_llm.quantization import QuantMode


def gen_suffix(rank, use_smooth_quant, quant_per_channel):
    suffix = f"{rank}.bin"
    if use_smooth_quant:
        sq_prefix = "int8."
        if quant_per_channel:
            sq_prefix += "col."
        suffix = sq_prefix + suffix
    return suffix


def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx])
    else:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])


def dup_kv_weight(v, num_head, tp_size):
    assert tp_size % num_head == 0
    reps = tp_size // num_head
    head_size = v.shape[0] // num_head
    v = v.reshape(num_head, head_size, -1)[:, None, :, :].expand(num_head, reps, head_size, v.shape[1])
    return v.reshape(num_head * reps * head_size, -1).clone().detach()

def load_from_hf_qwen2(tensorrt_llm_qwen2: QWenForCausalLM,
                      hf_qwen,
                      mapping=Mapping(),
                      #max_position_embeddings=32768,
                      #rotary_emb_base=10000,
                      #kv_channels=128,
                      dtype="float16",
                      #multi_query_mode=False,
                      #use_gemm_woq_plugin=False
                      ):
    tensorrt_llm.logger.info('Loading weights from HF QWen...')
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_qwen2, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()
    
    #num_kv_heads = tensorrt_llm_qwen2.config.num_key_value_heads
    #mha_mode = (num_kv_heads == tensorrt_llm_qwen2.num_attention_heads)
    mha_mode = True
    
    model_params = dict(hf_qwen.named_parameters())
    torch_dtype = str_dtype_to_torch(dtype)
    
    for k, v in tqdm(model_params.items(),
                     total=len(model_params),
                     ncols=80,
                     desc="Converting..."):
        
        #print(k)
        if 'visual' in k:
            continue
        if isinstance(v, list):
            v = [torch_to_numpy(vv.to(torch_dtype).detach().cpu()) for vv in v]
        else:
            v = torch_to_numpy(v.to(torch_dtype).detach().cpu())
            
        
        if 'embed_tokens.weight' in k:
            #weights['model.embedding.vocab_embedding.weight'] = v
            print("\n*********************************************************************************************************************************************")
            tensorrt_llm_qwen2.transformer.vocab_embedding.weight.value = v
        elif 'model.norm.weight' in k:
            #weights['model.norm.weight'] = v
            tensorrt_llm_qwen2.transformer.norm.weight.value = v
        elif 'lm_head.weight' in k:
            #weights['model.lm_head.weight'] = np.ascontiguousarray(split(v, mapping.tp_size, mapping.tp_rank))
            tensorrt_llm_qwen2.lm_head.weight.value = split(v, mapping.tp_size, mapping.tp_rank)
        else:
            layer_idx = extract_layer_idx(k)
            
            if layer_idx is None:
                continue
            idx = int(layer_idx)

            if idx >= tensorrt_llm_qwen2.config.num_hidden_layers:
                continue
            
            tllm_prex = f'model.layers.{idx}.'
            if tllm_prex+'input_layernorm.weight' in k:                
                tensorrt_llm_qwen2.transformer.layers[idx].input_layernorm.weight.value = v
            elif tllm_prex+'post_attention_layernorm.weight' in k:
                #weights[tllm_prex+'post_attention_layernorm.weight'] = v
                tensorrt_llm_qwen2.transformer.layers[idx].post_attention_layernorm.weight.value = v
            # attention.q_proj
            elif 'self_attn.q_proj.weight' in k:
                myWeight=[]
                myWeight.append(v)
            elif 'self_attn.q_proj.bias' in k:
                myBias=[]
                myBias.append(v)
            elif 'self_attn.k_proj.weight' in k:
                myWeight.append(v)
            elif 'self_attn.k_proj.bias' in k:
                myBias.append(v)
            elif 'self_attn.v_proj.weight' in k:
                myWeight.append(v) 
            elif 'self_attn.v_proj.bias' in k:
                myBias.append(v)   
                # weight
                dst = tensorrt_llm_qwen2.transformer.layers[idx].attention.qkv.weight
                #print(dst.name)
                if not mha_mode:
                    assert isinstance(myWeight, list) and len(myWeight) == 3
                    wq = split(myWeight[0], mapping.tp_size, mapping.tp_rank)
                    wk = split(myWeight[1], mapping.tp_size, mapping.tp_rank)
                    wv = split(myWeight[2], mapping.tp_size, mapping.tp_rank)
                    split_v =  split(np.concatenate((wq, wk, wv),dim=0),mapping.tp_size, mapping.tp_rank)
                else:
                    myWeight = np.concatenate(myWeight)
                                        
                    q_emb = myWeight.shape[0] // 3
                    model_emb = myWeight.shape[1]
                    myWeight = myWeight.reshape(3, q_emb, model_emb)
                    
                    split_v = split(myWeight, mapping.tp_size, mapping.tp_rank, dim=1)
                    split_v = split_v.reshape(3 * (q_emb // mapping.tp_size), model_emb)
                    print('here=',split_v.shape)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    (processed_torch_weights, torch_weight_scales) = \
                        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(torch.tensor(v), plugin_weight_only_quant_type)
                    
                    dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_qwen2.transformer.layers[idx].attention.qkv.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = split_v
                    
                # bias
                dst = tensorrt_llm_qwen2.transformer.layers[idx].attention.qkv.bias
                if not mha_mode:
                    assert isinstance(myBias, list) and len(myBias) == 3
                    bq = split(myBias[0], mapping.tp_size, mapping.tp_rank)
                    bk = split(myBias[1], mapping.tp_size, mapping.tp_rank)
                    bv = split(myBias[2], mapping.tp_size, mapping.tp_rank)
                    split_v =  np.concatenate((bq, bk, bv))
                else:
                    myBias = np.concatenate(myBias)
                    #print("myBias= ",myBias.shape)
                    q_emb = myBias.shape[0] // 3
                    #model_emb = myBias.shape[1]
                    myBias = myBias.reshape(3, q_emb)
                    split_v = split(myBias, mapping.tp_size, mapping.tp_rank, dim=1)
                    split_v = split_v.reshape(3 * (q_emb // mapping.tp_size))
                
                dst.value = split_v

            elif tllm_prex+'self_attn.o_proj.weight' in k:
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    processed_torch_weights, torch_weight_scales = \
                        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                            numpy_to_torch(v), plugin_weight_only_quant_type)
                    tensorrt_llm_qwen2.transformer.layers[idx].attention.dense.weight.value = processed_torch_weights.numpy()
                    tensorrt_llm_qwen2.transformer.layers[idx].attention.dense.per_channel_scale.value = torch_weight_scales.numpy()
                else:
                    tensorrt_llm_qwen2.transformer.layers[idx].attention.dense.weight.value = split_v

            elif tllm_prex+'mlp.up_proj.weight' in k:
                dst = tensorrt_llm_qwen2.transformer.layers[idx].mlp.gate.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=0)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    (
                        processed_torch_weights,
                        torch_weight_scales,
                    ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type
                    )
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_qwen2.transformer.layers[idx].mlp.gate.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = split_v

            elif tllm_prex+'mlp.down_proj.weight' in k:
                dst = tensorrt_llm_qwen2.transformer.layers[idx].mlp.proj.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=1)
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    (
                        processed_torch_weights,
                        torch_weight_scales,
                    ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type
                    )
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_qwen2.transformer.layers[idx].mlp.proj.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = split_v
            elif tllm_prex+'mlp.gate_proj.weight' in k:
                dst = tensorrt_llm_qwen2.transformer.layers[idx].mlp.fc.weight
                split_v = split(v, mapping.tp_size, mapping.tp_rank, dim=0)
                
                if use_weight_only:
                    v = np.ascontiguousarray(split_v.transpose())
                    (
                        processed_torch_weights,
                        torch_weight_scales,
                    ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                        torch.tensor(v), plugin_weight_only_quant_type
                    )
                    # workaround for trt not supporting int8 inputs in plugins currently
                    dst.value = processed_torch_weights.numpy()
                    scales = tensorrt_llm_qwen2.transformer.layers[idx].mlp.fc.per_channel_scale
                    scales.value = torch_weight_scales.numpy()
                else:
                    dst.value = split_v
            else:
                print("unknown key: ", k)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')

    return 

#------------------------------------------------------------------------------------------------------------
def load_from_awq_qwen(tensorrt_llm_qwen2: QWenForCausalLM,
                        quant_ckpt_path,
                        quantize_lm_head=False,
                        mapping=Mapping(),
                        dtype="float16",
                        ft_model_dir=None):
    tensorrt_llm.logger.info(
        'Loading weights from groupwise QWen1.5 checkpoint...')
    tik = time.time()

    if quant_ckpt_path.endswith(".pt"):
        awq_qwen = torch.load(quant_ckpt_path)
        
        awq_prefix = "model."
        awq_suffix_list = [
            ".weight",
            ".weight_quantizer._amax",
            ".input_quantizer._pre_quant_scale",
        ]
        awq_key_list = [
            "embed_tokens.weight",  # vocab_embedding
            "lm_head",  # lm_head
            "norm.weight",  # ln_f
            "self_attn.",  # attention.qkv
            "_proj",  # qkv suffix
            "self_attn.o_proj",  # attention.dense
            "mlp.up_proj",  # mlp.gate
            "mlp.down_proj",  # mlp.proj
            "mlp.gate_proj",  # mlp.fc
            "input_layernorm.weight",  # input_layernorm
            "post_attention_layernorm.weight",  # post_layernorm
        ]
        split_sym = "."

        def load(key):
            if "lm_head" in key:
                print(key)
                v = awq_qwen[key]
            else:
                print(awq_prefix)
                print(key)
                v = awq_qwen[awq_prefix + key]
            return v

        group_size = load("layers.0.self_attn.o_proj.weight").numel() // load(
            "layers.0.self_attn.o_proj.weight_quantizer._amax").numel()
    else:
        assert False, "Unsupported AWQ quantized checkpoint format"

    quant_mode = getattr(tensorrt_llm_qwen2, 'quant_mode', QuantMode(0))
    # Int8 KV cache
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()

    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm
    torch_dtype = str_dtype_to_torch(dtype)

    def fromfile(dir_path, name, shape=None, dtype=None):
        p = dir_path + '/' + name
        if Path(p).exists():
            t = np.fromfile(p, dtype=dtype)
            if shape is not None:
                t = t.reshape(shape)
            return t
        return None

    def torch_split(v, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            tensorrt_llm.logger.error(
                "Current weight shape is invalid for mapping.tp_size=" +
                str(mapping.tp_size))
            assert False, "Invalid TP size"
        return v.split(v.shape[dim] // mapping.tp_size,
                       dim=dim)[mapping.tp_rank]

    def AWQ_quantize_pack_preprocess(weight, scale):
        weight /= scale.repeat_interleave(group_size, dim=0)
        qweight_int8 = torch.clamp(torch.round(weight.cuda()).char(), -8, 7)
        int4_weight = preprocessor(packer(qweight_int8.cpu()), torch.quint4x2)
        return int4_weight.view(torch.float16).cpu().numpy()

    def process_and_assign_weight(mOp, v, tp_dim=0):
        weight = v[0].T.contiguous()
        print(weight.shape)
        [k, n] = weight.shape
        weight = torch_split(weight, tp_dim)
        amax = v[1].reshape((n, k // group_size)).T.contiguous()
        amax = torch_split(amax, tp_dim)
        pre_quant_scale = v[2].reshape((1, k))
        if tp_dim == 0:
            pre_quant_scale = torch_split(pre_quant_scale, 1)
        scale = amax / 8.0
        mOp.weight.value = AWQ_quantize_pack_preprocess(weight, scale)
        mOp.weights_scaling_factor.value = scale.to(torch_dtype).cpu().numpy()
        mOp.prequant_scaling_factor.value = pre_quant_scale.to(
            torch_dtype).cpu().numpy()

    def reSmooth_and_get_scale(weight, pre_quant_scale, avg_pre_quant_scale):
        # deSmooth and reSmooth
        [k, n] = weight.shape
        if quant_ckpt_path.endswith("pt"):
            # NPZ files are already re-smoothed
            weight *= pre_quant_scale.repeat((n, 1)).transpose(1,
                                                               0).contiguous()
            weight /= avg_pre_quant_scale.repeat(
                (n, 1)).transpose(1, 0).contiguous()

        # Get scale
        weight_t = weight.T.contiguous()
        weight_t = weight_t.reshape(n, k // group_size, group_size)
        weight_t = torch.abs(weight_t.reshape(-1, group_size))
        amax, idx = weight_t.max(1)
        amax = amax.reshape(n, k // group_size).T.contiguous()
        scale = amax / 8
        return weight, scale

    def process_and_assign_qkv_weight(prefix, mOp):
        q_weight = load(prefix + "q" + awq_key_list[4] +
                        awq_suffix_list[0]).T.contiguous()
        k_weight = load(prefix + "k" + awq_key_list[4] +
                        awq_suffix_list[0]).T.contiguous()
        v_weight = load(prefix + "v" + awq_key_list[4] +
                        awq_suffix_list[0]).T.contiguous()
        dim_k = q_weight.shape[0]
        q_weight = torch_split(q_weight, 1)
        k_weight = torch_split(k_weight, 1)
        v_weight = torch_split(v_weight, 1)
        q_pre_quant_scale = load(prefix + "q" + awq_key_list[4] +
                                 awq_suffix_list[2]).reshape((1, dim_k))
        k_pre_quant_scale = load(prefix + "k" + awq_key_list[4] +
                                 awq_suffix_list[2]).reshape((1, dim_k))
        v_pre_quant_scale = load(prefix + "v" + awq_key_list[4] +
                                 awq_suffix_list[2]).reshape((1, dim_k))
        qkv_pre_quant_scale = (q_pre_quant_scale + k_pre_quant_scale +
                               v_pre_quant_scale) / 3.0
        q_weight, q_scale = reSmooth_and_get_scale(q_weight, q_pre_quant_scale,
                                                   qkv_pre_quant_scale)
        k_weight, k_scale = reSmooth_and_get_scale(k_weight, k_pre_quant_scale,
                                                   qkv_pre_quant_scale)
        v_weight, v_scale = reSmooth_and_get_scale(v_weight, v_pre_quant_scale,
                                                   qkv_pre_quant_scale)
        qkv_weights = torch.cat((q_weight, k_weight, v_weight), dim=1)
        qkv_scale = torch.cat((q_scale, k_scale, v_scale), dim=1)

        #print(mOp)
        mOp.prequant_scaling_factor.value = qkv_pre_quant_scale.to(torch_dtype).cpu().numpy()
        mOp.weight.value = AWQ_quantize_pack_preprocess(qkv_weights, qkv_scale)
        mOp.weights_scaling_factor.value = qkv_scale.to(torch_dtype).cpu().numpy()

    def process_and_assign_weight_for_lm_head(mPrefix, mOp, tp_dim=0):
        weight = load(mPrefix + ".weight").T.contiguous()
        [k, n] = weight.shape
        weight = torch_split(weight, tp_dim)
        amax = load(mPrefix + ".weight_quantizer._amax").reshape(
            (n, int(k / group_size))).T.contiguous()
        amax = torch_split(amax, tp_dim)
        pre_quant_scale = load(mPrefix + ".input_quantizer._pre_quant_scale").reshape((1, k))
        if tp_dim == 0:
            pre_quant_scale = torch_split(pre_quant_scale, 1)
        scale = amax / 8.0
        mOp.weight.value = AWQ_quantize_pack_preprocess(weight, scale)
        mOp.weights_scaling_factor.value = scale.to(torch_dtype).cpu().numpy()
        mOp.prequant_scaling_factor.value = pre_quant_scale.to(
            torch_dtype).cpu().numpy()

    # Load weights from AWQ checkpoint into TRT-LLM module
    # 1. vocab_embedding
    # Check if we need to pad vocab
    v = load(awq_key_list[0])
    [vocab_size, k] = v.shape
    pad_vocab = False
    pad_vocab_size1 = vocab_size
    if quantize_lm_head and vocab_size % 64 != 0:
        pad_vocab = True
        pad_vocab_size1 = int((vocab_size + 63) / 64) * 64
    if pad_vocab:
        new_v = torch.zeros([pad_vocab_size1, k])
        new_v[:vocab_size, :] = v
        v = new_v
    if mapping.is_first_pp_rank():
        tensorrt_llm_qwen2.transformer.vocab_embedding.weight.value = v.to(
            torch_dtype).cpu().numpy()

    # 2. lm_head        
    if pad_vocab:
        weight = load['lm_head.weight']
        [vocab_size, k] = weight.shape
        new_weight = torch.zeros([pad_vocab_size1, k])
        new_weight[:vocab_size, :] = weight
        new_weight = new_weight.T.contiguous()
        amax = load['lm_head.weight_quantizer._amax'].reshape(
            [vocab_size, k // group_size])
        new_amax = torch.ones([pad_vocab_size1, k // group_size])
        new_amax[:vocab_size, :] = amax
        new_amax = new_amax.T.contiguous()
        new_scale = new_amax / 8
        tensorrt_llm_qwen2.lm_head.weight.value = AWQ_quantize_pack_preprocess(
            new_weight, new_scale)
        tensorrt_llm_qwen2.lm_head.weights_scaling_factor.value = new_scale.to(
            torch_dtype).cpu().numpy()
        tensorrt_llm_qwen2.lm_head.prequant_scaling_factor.value = load[
            'lm_head.input_quantizer._pre_quant_scale'].to(
                torch_dtype).cpu().numpy()
    elif quantize_lm_head:
        mOp = tensorrt_llm_qwen2.lm_head
        if mapping.is_last_pp_rank():
            process_and_assign_weight_for_lm_head('lm_head',mOp,1)
    else:
        tensorrt_llm_qwen2.lm_head.weight.value = torch_split(
            load('lm_head.weight'), 0).to(torch_dtype).cpu().numpy()
    
  
    # 3. ln_f
    v = load(awq_key_list[2])
    if mapping.is_last_pp_rank():
        tensorrt_llm_qwen2.transformer.norm.weight.value = v.to(torch_dtype).cpu().numpy()

    # 4. Weights inside each layer
    num_hidden_layers = tensorrt_llm_qwen2.config.num_hidden_layers
    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))

    for l in layers_range:
        layer_idx = l - mapping.pp_rank * layers_per_pipeline_stage
        prefix = "layers" + split_sym + str(layer_idx) + split_sym
        tensorrt_llm.logger.info(f'Process weights in layer: {layer_idx}')
        layer = tensorrt_llm_qwen2.transformer.layers[layer_idx]

        # 4.1 attention.qkv
        process_and_assign_qkv_weight(prefix + awq_key_list[3],
                                      layer.attention.qkv)
        
        # 4.1.2 attention.qkv.bias
        qkv_bias_list = []
        for x in ["q", "k", "v"]:
            x_bias = load(prefix + f"self_attn.{x}_proj.bias")
            x_bias = torch_split(x_bias, dim=0)
            qkv_bias_list.append(x_bias)        
        qkv_bias = torch.cat(qkv_bias_list, dim=0)
        split_v = split(qkv_bias, mapping.tp_size, mapping.rank, dim=1)
        split_v = split_v.reshape(3 * ((qkv_bias.shape[0]//3)// mapping.tp_size))
        layer.attention.qkv.bias.value = np.ascontiguousarray(
            qkv_bias.to(torch_dtype).cpu().numpy()
        )

        # 4.2 attention.dense
        v = [load(prefix + awq_key_list[5] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.attention.dense, v, 0)

        # 4.3 mlp.gate
        v = [load(prefix + awq_key_list[6] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.mlp.gate, v, 1)

        # 4.4 mlp.proj
        v = [load(prefix + awq_key_list[7] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.mlp.proj, v, 0)

        # 4.5 mlp.fc
        v = [load(prefix + awq_key_list[8] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.mlp.fc, v, 1)

        # 4.6 input_layernorm
        v = load(prefix + awq_key_list[9])
        layer.input_layernorm.weight.value = v.to(torch_dtype).cpu().numpy()

        # 4.7 post_layernorm
        v = load(prefix + awq_key_list[10])
        layer.post_attention_layernorm.weight.value = v.to(torch_dtype).cpu().numpy()

        # 4.8 attention.kv_quant_orig_scale / kv_quant_orig_scale
        if use_int8_kv_cache:
            assert ft_model_dir, "You must pass --ft_model_dir to tell TRT-LLM where to look for scales of INT8 kv cache."
            t = fromfile(
                ft_model_dir, 'model.layers.' + str(layer_idx) +
                '.attention.query_key_value.scale_y_quant_orig.bin', [1],
                np.float32)
            assert t is not None, f"{ft_model_dir} does not contain model.layers.{layer_idx}.attention.query_key_value.scale_y_quant_orig.bin"
            layer.attention.kv_orig_quant_scale.value = 1.0 / t
            layer.attention.kv_quant_orig_scale.value = t

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
