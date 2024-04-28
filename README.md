# QWen1.5_TensorRT-LLM
In this branch, we will optimize QWen1.5 with Tensorrt-LLM-0.8.0.

Qwen1.5 is the beta version of Qwen2. Its network structure is different from Qwen1.

Currently, we have realized FP16 and INT8/INT4 Weight-Only, INT8/INT4-Weight-Only-AWQ for Qwen1.5. 

For this repo, we aim to realize the model by using as more as possible predefined layer and functions in TRTLLM, so that to leverage more features(FP8,quantization,...) provided by TRTLLM.


## Getting started

We developed and tested them inside TRTLLM-0.8.0 container.

Before we go, if the existed "transformers" version < 4.38.1, we should upgrade the "transformers" library to the latest with:
```
pip3 install --upgrade transformers
```

To use this code, please put the all the "*.py" files into a same folder (we assume this folder named as "For0.8.0").

By the way, dataset of "cnn_dailymail" could also be downloaded from "[https://gitee.com/hf-datasets/cnn_dailymail/tree/script/](https://gitee.com/hf-datasets/cnn_dailymail/tree/script/)". And we should modify "_DL_URLS" in "cnn_dailymail.py" with proper http links or local paths like below:
```
_DL_URLS = {
    "cnn_stories": "/path_to/cnn_dailymail/data/cnn_stories.tgz",
    "dm_stories": "/path_to/cnn_dailymail/data/dailymail_stories.tgz",
    "train": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_train.txt",
    "validation": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_val.txt",
    "test": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_test.txt",
}
```

Then, modify the line 199 as below:
```
from
    dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
to
    dataset = load_dataset("/path_to/cnn_dailymail/cnn_dailymail.py", name="3.0.0", split="train")
```
### Build the FP16 engine

- Need to prepare the HF Qwen checkpoint by following the guides here Qwen-7B-Chat
TensorRT-LLM builds TensorRT engine(s) from HF checkpoint. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) with dummy weights.
Normally trtllm-build only requires single GPU, but if you've already got all the GPUs needed while inferencing, you could enable parallelly building to make the engine building process faster by adding --workers argument. Please note that currently workers feature only supports single node.
Here're some examples:

```
# Build a single-GPU float16 engine from HF weights.
# Try use_gemm_plugin to prevent accuracy issue.

# Build the Qwen-7B-Chat model using a single GPU and FP16.

python /Qwen2-7B/For0.8.0/build.py --model_dir /Qwen2-7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
		--world_size 1	\
		--tp_size 1	\
		--pp_size 1 \
                --output_dir /Qwen2-7B/trt_engines/float16/1-gpu/


# Build the Qwen-7B-Chat model using 4 GPUs and FP16.

python /Qwen2-7B/For0.8.0/build.py --model_dir /Qwen2-7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
		--world_size 4	\
		--tp_size 4	\
		--pp_size 1 \
                --output_dir /Qwen2-7B/trt_engines/float16/4-gpu/
```
#### Run the engines
- To run the engine with TP=1
```
python /app/tensorrt_llm/examples/run.py    	--tokenizer_dir /Qwen2-7B \
                                            	--engine_dir /Qwen2-7B/trt_engines/float16/1-gpu/ \
                                            	--max_output_len 50 \
						--input_text "中国的首都是哪个城市？"
```

- To run the engine with TP=4
```
mpirun -n 4 --allow-run-as-root \
python /app/tensorrt_llm/examples/run.py    	--tokenizer_dir /Qwen2-7B \
                                            	--engine_dir /Qwen2-7B/trt_engines/float16/4-gpu/ \
                                            	--max_output_len 50 \
						--input_text "中国的首都是哪个城市？"
```

### INT4-WO-AWQ

#Quantize the HF model
```
python /app/tensorrt_llm/examples/quantization/quantize.py 	--model_dir /Qwen2-7B \
						--batch_size 1 \
                                    		--dtype float16 \
                                    		--qformat int4_awq \                                    
                                    		--calib_size 32   \
						--tp_size 1	\
						--pp_size 1	\
						--output_dir /Qwen2-7B/For0.8.0/exported_model.pt
```
#Building the engine with quantized checkpoint
```
python build.py --model_dir /Qwen2-7B \
		--max_batch_size 1 \
                --quant_ckpt_path /Qwen2-7B/For0.8.0/exported_model.pt \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4_awq \
                --per_group \
                --world_size 1 \
                --tp_size 1 \
                --output_dir /Qwen2-7B/For0.8.0/trt_engines/int4-awq/1-gpu
```
#run the engine
```
python /app/tensorrt_llm/examples/run.py    	--tokenizer_dir /Qwen2-7B \
                                            	--engine_dir /Qwen2-7B/For0.8.0/trt_engines/int4-awq/1-gpu \
                                            	--max_output_len 50 \
						--input_text "中国的首都是哪里？"
```

#### Check the accuracy of the optimized engine
```
python /Qwen2-7B/For0.8.0/summarize.py  --test_trt_llm        \
                                        --test_hf        \
                                        --hf_model_dir /Qwen2-7B/   \
                                        --data_type fp16 \
                                        --engine_dir /Qwen2-7B/trt_engines/float16/1-gpu/  \
                                        --check_accuracy
```
### INT4-WO
```
# Build the engine of INT4-weight-only with TP=4
python /Qwen2-7B/For0.8.0/build.py --model_dir /Qwen2-7B \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
		--use_weight_only \
		--weight_only_precision int4	\
		--world_size 4	\
		--tp_size 4	\
		--pp_size 1 \
                --output_dir /Qwen2-7B/trt_engines/int4-wo/4-gpu/
```
