# QWen1.5_TensorRT-LLM
We try to optimize Qwen1.5 (7B,14B,72B) with Tensorrt-LLM-0.9.0.

Qwen1.5 is the beta version of Qwen2. Its network structure is different from Qwen1.

Currently, we have realized FP16 and INT8/INT4 Weight-Only, FP8, INT8/INT4-Weight-Only-AWQ for Qwen1.5. 

For this repo, we aim to realize the model by using as more as possible predefined layer and functions in TRTLLM, so that to leverage more features(FP8,quantization,...) provided by TRTLLM.


## Getting started

We developed and tested them inside TRTLLM-0.9.0 container.

Before we go, if the existed "transformers" version < 4.38.1, we should upgrade the "transformers" library to the latest with:
```
pip3 install --upgrade transformers
```

To use this code, please put the "convert_checkpoint.py" into the same folder.

And replace "/usr/local/lib/python3.10/dist-packages/tensorrt_llm/models/qwen/model.py" with this "model.py"

And replace "/app/tensorrt_llm/examples/utils.py" with "utils.py" in this repo.

And replace "/app/tensorrt_llm/examples/qwen/convert_checkpoint.py" with this "convert_checkpoint.py".

If NVIDIA-ammo-0.7.x installed, please replace "/usr/local/lib/python3.10/dist-packages/ammo/torch/export/layer_utils.py" with this "layer_utils.py". 

If NVIDIA-ammo-0.9.x installed, please modify "/usr/local/lib/python3.10/dist-packages/tensorrt_llm/quantization/quantize_by_ammo.py" as below:
```
Inside function of "quantize_and_export", modify
    from ammo.torch.export import export_model_config
to
    from ammo.torch.export.model_config_export import export_tensorrt_llm_checkpoint
and replace "export_model_config" as "export_tensorrt_llm_checkpoint",
and delete the line of "export_tensorrt_llm_config=(not export_npz),"
```
By the way, dataset of "cnn_dailymail" could also be downloaded from "https://gitee.com/hf-datasets/cnn_dailymail/tree/main". And we should modify "_DL_URLS" in "cnn_dailymail.py" with proper http links or local paths. Then, modify the line 199 as below:
```
from
    dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
to
    dataset = load_dataset("/path_to/cnn_dailymail/cnn_dailymail.py", name="3.0.0", split="train")
```
## Build the engine

- Need to prepare the HF Qwen checkpoint by following the guides here Qwen-7B-Chat
TensorRT-LLM builds TensorRT engine(s) from HF checkpoint. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) with dummy weights.
Normally trtllm-build only requires single GPU, but if you've already got all the GPUs needed while inferencing, you could enable parallelly building to make the engine building process faster by adding --workers argument. Please note that currently workers feature only supports single node.
Here're some examples:

```
# Build a single-GPU float16 engine from HF weights.
# Try use_gemm_plugin to prevent accuracy issue.

# Build the Qwen-7B-Chat model using a single GPU and FP16.
python convert_checkpoint.py --model_dir ./tmp/Qwen/7B/ \
                              --output_dir ./tllm_checkpoint_1gpu_fp16 \
                              --dtype float16

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16 \
            --output_dir ./tmp/qwen/7B/trt_engines/fp16/1-gpu \
            --gemm_plugin float16

# Build the Qwen-7B-Chat model using a single GPU and BF16.
python convert_checkpoint.py --model_dir ./tmp/Qwen/7B/ \
                              --output_dir ./tllm_checkpoint_1gpu_bf16 \
                              --dtype bfloat16

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_bf16 \
            --output_dir ./tmp/qwen/7B/trt_engines/bf16/1-gpu \
            --gpt_attention_plugin bfloat16 \
            --gemm_plugin bfloat16

# Build the Qwen-7B-Chat model using a single GPU and apply INT8 weight-only quantization.
python convert_checkpoint.py --model_dir ./tmp/Qwen/7B/ \
                              --output_dir ./tllm_checkpoint_1gpu_fp16_wq \
                              --dtype float16 \
                              --use_weight_only \
                              --weight_only_precision int8

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16_wq \
            --output_dir ./tmp/qwen/7B/trt_engines/weight_only/1-gpu/ \
            --gemm_plugin float16

# Build the Qwen-7B-Chat model using a single GPU and apply INT4 weight-only quantization.
python convert_checkpoint.py --model_dir ./tmp/Qwen/7B/ \
                              --output_dir ./tllm_checkpoint_1gpu_fp16_wq \
                              --dtype float16 \
                              --use_weight_only \
                              --weight_only_precision int4

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16_wq \
            --output_dir ./tmp/qwen/7B/trt_engines/weight_only/1-gpu/ \
            --gemm_plugin float16

# Build Qwen-7B-Chat using 2-way tensor parallelism.
python convert_checkpoint.py --model_dir ./tmp/Qwen/7B/ \
                            --output_dir ./tllm_checkpoint_2gpu_tp2 \
                            --dtype float16 \
                            --tp_size 2

trtllm-build --checkpoint_dir ./tllm_checkpoint_2gpu_tp2 \
            --output_dir ./tmp/qwen/7B/trt_engines/fp16/2-gpu/ \
            --gemm_plugin float16

# Build Qwen-7B-Chat using 2-way tensor parallelism and 2-way pipeline parallelism.
python convert_checkpoint.py --model_dir ./tmp/Qwen/7B/ \
                            --output_dir ./tllm_checkpoint_4gpu_tp2_pp2 \
                            --dtype float16 \
                            --tp_size 2 \
                            --pp_size 2
trtllm-build --checkpoint_dir ./tllm_checkpoint_4gpu_tp2_pp2 \
            --output_dir ./tmp/qwen/7B/trt_engines/fp16/4-gpu/ \
            --gemm_plugin float16

# Build Qwen-14B-Chat using 2-way tensor parallelism.
python convert_checkpoint.py --model_dir ./tmp/Qwen/14B/ \
                            --output_dir ./tllm_checkpoint_2gpu_tp2 \
                            --dtype float16 \
                            --tp_size 2

trtllm-build --checkpoint_dir ./tllm_checkpoint_2gpu_tp2 \
            --output_dir ./tmp/qwen/14B/trt_engines/fp16/2-gpu/ \
            --gemm_plugin float16 \

# Build Qwen-72B-Chat using 8-way tensor parallelism.
python convert_checkpoint.py --model_dir ./tmp/Qwen/72B/ \
                            --output_dir ./tllm_checkpoint_8gpu_tp8 \
                            --dtype float16 \
                            --tp_size 8

trtllm-build --checkpoint_dir ./tllm_checkpoint_8gpu_tp8 \
            --output_dir ./tmp/qwen/72B/trt_engines/fp16/8-gpu/ \
            --gemm_plugin float16 \


```

## Run the engines
- To run the engine with TP=1
```
python /app/tensorrt_llm/examples/run.py    --tokenizer_dir /path_to/Qwen1.5-7B-Chat \
                                            --engine_dir /path_to/Qwen1.5-7B-Chat/trt_engines/float16/1-gpu/ \ 
                                            --max_output_len 50
```


- To run the engine with TP=4
```
mpirun -n 2 --allow-run-as-root \
python /app/tensorrt_llm/examples/run.py    --tokenizer_dir /path_to/Qwen1.5-7B-Chat \
                                            --engine_dir /path_to/Qwen1.5-7B-Chat/trt_engines/int4_weight_only/4-gpu/ \
                                            --max_output_len 50
```

## FP8 Post-Training Quantization
```
python ./quantization/quantize.py --model_dir /path_to/Qwen2-7B/ \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir /path_to/Qwen2-7B/quantized_fp8 \
                                   --calib_size 32

trtllm-build --checkpoint_dir /path_to/Qwen2-7B/quantized_fp8 \
             --output_dir /path_to/Qwen2-7B/engine_outputs \
             --gemm_plugin float16 \
             --strongly_typed \
             --workers 1

python3 ./run.py --max_output_len=50 \
                  --tokenizer_dir /path_to/Qwen2-7B/ \
                  --engine_dir=/path_to/Qwen2-7B/engine_outputs/

```

## INT4-WO-AWQ
```
#Quantize the HF model
python3 ./quantization/quantize.py  --model_dir /path_to/Qwen2-7B \
                                    --dtype float16 \
                                    --qformat int4_awq \
                                    --output_dir /path_to/Qwen2-7B/qwen_7b_INT4_WO_AWQ \
                                    --calib_size 32

#Building the engine with quantized checkpoint
trtllm-build --checkpoint_dir /path_to/Qwen2-7B/qwen_7b_INT4_WO_AWQ \
             --output_dir /path_to/Qwen2-7B/trt_engines/int4_AWQ/1-gpu/ \
             --gemm_plugin float16

#run the engine
python3 ./run.py --max_output_len=50 \
                  --tokenizer_dir /path_to/Qwen2-7B/ \
                  --engine_dir=/path_to/Qwen2-7B/trt_engines/int4_AWQ/1-gpu/
```

## Check the accuracy of the optimized engine
```
python /app/tensorrt_llm/examples/summarize.py  --test_trt_llm      \                  
                                                --hf_model_dir /path_to/Qwen2-7B/       \                 
                                                --data_type fp16      \                  
                                                --engine_dir /path_to/Qwen2-7B/trt_engines/int4_AWQ/1-gpu/      \                  
                                                --max_input_length 1024       \                 
                                                --output_len 1024       \                  
                                                --check_accuracy

```
Finally, we will get the Rouge result of int4-AWQ like below (on A100-80GB):
```
[04/08/2024-16:16:33] [TRT-LLM] [I] ---------------------------------------------------------
[04/08/2024-16:16:45] [TRT-LLM] [I] TensorRT-LLM (total latency: 11.459680318832397 sec)
[04/08/2024-16:16:45] [TRT-LLM] [I] TensorRT-LLM (total output tokens: 1556)
[04/08/2024-16:16:45] [TRT-LLM] [I] TensorRT-LLM (tokens per second: 135.7804019578914)
[04/08/2024-16:16:45] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[04/08/2024-16:16:45] [TRT-LLM] [I]   rouge1 : 25.406849748838066
[04/08/2024-16:16:45] [TRT-LLM] [I]   rouge2 : 7.29327996818365
[04/08/2024-16:16:45] [TRT-LLM] [I]   rougeL : 17.76100218929106
[04/08/2024-16:16:45] [TRT-LLM] [I]   rougeLsum : 21.45597579324207
```

As for reference, native FP16 engine under the same test get the Rouge result like below (on A100-80GB), which is a quite comparative result.
```
[04/09/2024-09:22:41] [TRT-LLM] [I] ---------------------------------------------------------
[04/09/2024-09:23:15] [TRT-LLM] [I] TensorRT-LLM (total latency: 32.88514947891235 sec)
[04/09/2024-09:23:15] [TRT-LLM] [I] TensorRT-LLM (total output tokens: 2835)
[04/09/2024-09:23:15] [TRT-LLM] [I] TensorRT-LLM (tokens per second: 86.20912615337046)
[04/09/2024-09:23:15] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[04/09/2024-09:23:15] [TRT-LLM] [I]   rouge1 : 22.76706353606526
[04/09/2024-09:23:15] [TRT-LLM] [I]   rouge2 : 6.16967158340081
[04/09/2024-09:23:15] [TRT-LLM] [I]   rougeL : 15.752450766826765
[04/09/2024-09:23:15] [TRT-LLM] [I]   rougeLsum : 18.320161815021883
```
