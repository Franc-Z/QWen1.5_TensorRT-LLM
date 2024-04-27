# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional
from transformers.utils import logging

import tensorrt as trt

from tensorrt_llm._utils import pad_vocab_size
from tensorrt_llm.functional import (AttentionMaskType, ACT2FN, Tensor, send, recv, partial,unary)
from tensorrt_llm.layers import (Attention, AttentionMaskType,  ColumnLinear, 
                                Embedding, GatedMLP, RmsNorm)
from tensorrt_llm.module import Module

from tensorrt_llm.models.modeling_utils import (DecoderLayerList, DecoderModelForCausalLM, PretrainedConfig)


log = partial(unary, op=trt.UnaryOperation.LOG)
ceil = partial(unary, op=trt.UnaryOperation.CEIL)


logger = logging.get_logger(__name__)


 
class QWenDecoderLayer(Module):

    def __init__(self,
                 config: PretrainedConfig, 
                 layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        dtype = config.dtype
        tp_group = config.mapping.tp_group
        tp_size = config.mapping.tp_size

        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]


        self.attention = Attention(
            #local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,       
            max_position_embeddings=config.max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            position_embedding_type=config.position_embedding_type,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            tp_group=tp_group,
            tp_size=tp_size,
            quant_mode=config.quant_mode,
            dense_bias=False)
        
        

        self.mlp = GatedMLP(hidden_size=config.hidden_size,
                          ffn_hidden_size=config.intermediate_size, 
                          hidden_act=config.hidden_act,
                          dtype=dtype,
                          bias=False,
                          tp_group=tp_group,
                          tp_size=tp_size,
                          quant_mode=config.quant_mode,
                          )
        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                            eps=config.norm_epsilon,
                            dtype=dtype)
        self.post_attention_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                            eps=config.norm_epsilon,
                            dtype=dtype)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask = None,
        use_cache=True,
        kv_cache_params=None,
        attention_params=None,
    ):
        #https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/qwen2/modeling_qwen2.py
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        #print(kv_cache_params.is_valid(default_net().plugin_config.gpt_attention_plugin))
                     
        #assert kv_cache_params.is_valid(default_net().plugin_config.gpt_attention_plugin) or kv_cache_params is None
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )
        #print("attention_output =",attention_output)
        

        if use_cache:
            attention_output, present_key_value = attention_output

        self.register_network_output('attention_output', attention_output)
        hidden_states = residual + attention_output

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        #for debug
        #self.register_network_output('hidden_states', hidden_states)
        
        if use_cache:
            return (hidden_states, present_key_value)
        return hidden_states


class QWenModel(Module):
    def __init__(
        self,
        config: PretrainedConfig
    ):
        super().__init__()
        self.mapping = config.mapping
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(
                config.vocab_size,
                config.hidden_size,
                dtype=config.dtype,
                tp_size=self.mapping.tp_size
                if config.use_parallel_embedding else 1,
                tp_group=self.mapping.tp_group
                if config.use_parallel_embedding else None,
                sharding_dim=config.embedding_sharding_dim,
                tp_rank=self.mapping.tp_rank)


        self.layers = DecoderLayerList(QWenDecoderLayer, config)

        if self.mapping.is_last_pp_rank():
            self.norm = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)

    def forward(self,
                input_ids: Tensor=None,
                position_ids: Tensor=None,
                use_cache=True,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                prompt_embedding_table: Optional[Tensor]=None,
                prompt_tasks: Optional[Tensor]=None,
                prompt_vocab_size: Optional[Tensor] = None):

        #kv_cache_params.fill_none_tensor_list(len(self.layers))
       
        if use_cache:
            presents = []

        ptuning_args = [
            prompt_embedding_table, prompt_tasks, prompt_vocab_size
        ] if prompt_embedding_table is not None else []

        if self.mapping.is_first_pp_rank():
            print('here is',type(input_ids))
            hidden_states = self.vocab_embedding(input_ids,*ptuning_args)            
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        hidden_states = self.layers.forward(hidden_states,
                                            use_cache=use_cache,
                                            attention_mask=attention_mask,
                                            kv_cache_params=kv_cache_params,
                                            attention_params=attention_params)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class QWenForCausalLM(DecoderModelForCausalLM):

    def __init__(
        self,
        config:PretrainedConfig            
    ):
        self.check_config(config)
        transformer = QWenModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,config.mapping.tp_size)
         
        if config.mapping.is_last_pp_rank():
            lm_head = ColumnLinear(config.hidden_size,
                                    vocab_size_padded,
                                    bias=False,
                                    dtype=config.dtype,
                                    tp_group=config.mapping.tp_group,
                                    tp_size=config.mapping.tp_size,
                                    gather_output=True)
        else:
            lm_head = None
        
        self.quant_mode = config.quant_mode
        self.mapping = config.mapping
        super().__init__(config, transformer, lm_head)

    def check_config(self, config):
        config.set_if_not_exist('rotary_base', 1000000.0)
        config.set_if_not_exist('rotary_scaling', None)
