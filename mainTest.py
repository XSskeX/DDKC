from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from generate import Qwen2
from tokenizer import Tokenizer
from typing import List, Optional, Tuple
import os
import json
import time
from pathlib import Path
from typing import List, Optional
from safetensors.torch import load_file
import torch
from tokenizer import Dialog
import torch.nn.functional as F
model_name_or_path = 'C:/Users/HP/Desktop/qwen/qwenModel'
tokenizer = Tokenizer(model_path='C:/Users/HP/Desktop/qwen/qwenModel/tokenizer.model')
qwen2 = Qwen2.build(
    ckpt_dir="C:/Users/HP/Desktop/qwen/qwenModel",
    max_seq_len=2048,
    tokenizer=tokenizer,
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
prompt = "你好"

# 调用生成函数
generated_ids = qwen2.generate(
    tokenizer=tokenizer,
    prompt=prompt,
    max_length=50,
    temperature=0.1,
    top_k=1,
    top_p=0.01,
    device="cuda",
)
output_text = generated_ids
print(output_text)


