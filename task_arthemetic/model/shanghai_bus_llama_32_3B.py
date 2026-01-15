import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import unsloth


from datasets import load_dataset
from unsloth import FastLanguageModel, add_new_tokens
from unsloth import UnslothTrainer, UnslothTrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, IA3Config, LoraConfig, prepare_model_for_kbit_training, PeftModel, PeftConfig
import torch
from datasets import load_dataset, Dataset, DatasetDict
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import time
import numpy as np
from huggingface_hub import login
import argparse
import json
import ast
from transformers import TrainerCallback
import time

def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)
    
class LossPrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 只在主进程打印，且仅当有 loss 时
        if not state.is_world_process_zero or not logs:
            return
        if "loss" in logs:
            lr = logs.get("learning_rate", float("nan"))
            tqdm.write(f"[{time.strftime('%m-%d %H:%M:%S')}] "
                       f"step {state.global_step:,} | loss {logs['loss']:.4f} | lr {lr:.2e}")



if True:
    #config_params = load_config('./config/shanghai_bus_config.json')
    config_params = load_config('./config/shanghai_bus_qwen.json')
    # Basic Training Paramters
    traj_file_path= config_params.get('traj_file_path')
    # 轨迹文件路径（文本，每行一条训练样本）
    token = config_params.get('token')
    # HuggingFace token（用于加载私有模型、验证身份）
    model_name = config_params.get('model_name')
    # 模型名称
    batch_size = config_params.get('batch_size')
    lr = config_params.get('lr')
    lora_alpha = config_params.get('lora_alpha')
    lora_dropout = config_params.get('lora_dropout')
    lora_r = config_params.get('lora_r')
    device = config_params.get('device')
    MAX_LENGTH = config_params.get('MAX_LENGTH')
    num_epochs = config_params.get('num_epochs')
    save_model_dir = config_params.get('save_model_dir')
    ## 模型保存目录（将保存 LoRA adapter + tokenizer）
    h3_file_dir_origin=config_params.get('h3_file_dir_origin')
    # 源城市 H3 索引列表文件路径（每行一个 H3 格子索引）
    h3_file_dir_target=config_params.get('h3_file_dir_target')
    # 目标城市 H3 索引列表文件路径（每行一个 H3 格子索引）
    data_type=config_params.get('task_type')
    # 当前任务类型：'bus' 表示训练公交轨迹；'real' 表示训练真实轨迹
    real_lambda=config_params.get('real_lambda')
    #对于二阶邻居，聚合的时候embedding加权的系数
    bus_model_dir = config_params.get("bus_model_dir")
    # 加载已训练公交模型的目录（用于 real 阶段迁移 + 拷贝 H3 embedding）
    neighbor_init_map_file=config_params.get("neighbor_init_map_file")
    # 对于公交轨迹中不覆盖的h3index，用哪些邻居加权的字典




#################################读取轨迹文件######################################
prompts = []

# 打开轨迹文件，每一行是一条轨迹（用字符串表示的列表）
with open(traj_file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        # 去掉换行符和前后空格

        prompts.append(line)
print(prompts[0])
#######################################################################





##################################加载模型############################################################################################################
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = MAX_LENGTH,
    dtype = None,            # None = 自动推断精度
    load_in_4bit = True,     # 默认用 4bit 节省显存
    token = token,           # 登录用的 token
)
tokenizer.pad_token_id = tokenizer.bos_token_id
##############################################################################################################################################



####################################添加新token##########################################################################################################
with open(h3_file_dir_origin, "r") as f_origin:
        h3_tokens_origin = [line.strip() for line in f_origin]
# 再读取目标城市 H3 索引列表
with open(h3_file_dir_target, "r") as f_target:
        h3_tokens_target = [line.strip() for line in f_target]
# 拼接为一个总的 h3_tokens 列表
h3_tokens = h3_tokens_origin + h3_tokens_target
#得到一个 list，h3_tokens = ["<H3_8928308280fffff>", "<H3_8928308280aaaaf>", ...]

# 加到 tokenizer
add_new_tokens(model, tokenizer, new_tokens=h3_tokens)
##############################################################################################################################################


for tok_id in range(model.get_input_embeddings().num_embeddings):
     tok = tokenizer.convert_ids_to_tokens(tok_id)
     if tok not in h3_tokens:
            model.get_input_embeddings().weight.requires_grad[tok_id] = False
            model.get_output_embeddings().weight.requires_grad[tok_id] = False


#######################################添加LoRA#######################################################################################################
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_r,
    lora_alpha = lora_alpha,
    lora_dropout = lora_dropout,
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],  # GeoLLaMA 核心
    bias = "none"
    #现在要做的是 因果语言建模 (Causal Language Modeling, CLM)
    #会决定 forward 里调用 labels 时的行为（shift logits by 1）
)
##############################################################################################################################################################




########################################################################unsloth的数据集准备
prompts_with_eos = [p + tokenizer.eos_token for p in prompts]
dataset = Dataset.from_dict({"text": prompts_with_eos})
train_dataset, test_dataset = dataset.train_test_split(test_size=0.2, seed=1215).values()
########################################################################################################################################

print(train_dataset[0])
##############################################################################################################################################


##################################### Trainer 设置 ######################################
trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    dataset_text_field = "text",   # 直接用 text 字段
    max_seq_length = MAX_LENGTH,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = 8,
        num_train_epochs = num_epochs,
        warmup_ratio = 0.1,
        #max_steps = 2,                     
        #warmup_ratio = 0.0, 
        logging_strategy = "steps",
        learning_rate = lr,
        logging_steps = 10,
        save_strategy = "epoch",
        output_dir = save_model_dir,
        bf16 = True,   # 如果是 A100/H100 建议开 bf16
        optim = "adamw_8bit",  # 默认更高效的优化器
        weight_decay = 0.01,
        report_to = "none",  
    ),
)

trainer.add_callback(LossPrinterCallback())
##############################################################################################################################################


# ==============================
# 训练前：记录 GPU 初始状态
# ==============================
gpu_stats = torch.cuda.get_device_properties(0)
print(gpu_stats)
#打印 GPU 型号 & 总显存；
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
#拿到训练过程中的峰值显存
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved before training.")

trainer_stats =trainer.train()

# ==============================
# 训练后：统计 GPU 显存 + 时间
# ==============================
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# ==============================
# 保存模型 + tokenizer
# ==============================
trainer.save_model(save_model_dir)         
'''
在你用 Unsloth/PEFT (LoRA) 的情况下，它默认保存的是：

LoRA adapter 权重（即低秩矩阵 A、B）；

embedding 层里新增的 H3 token embedding；

其他被冻结的原始 LLM 权重不会保存（需要时还是从 from_pretrained(model_name) 重新加载）
'''
tokenizer.save_pretrained(save_model_dir)  # 保存 tokenizer（包含新加的 H3 tokens）
'''
这个会额外保存：

vocab.json / tokenizer.json / special_tokens_map.json / tokenizer_config.json 等文件；

里面包含你新增的 <H3_...> token；

这样 real 阶段加载的时候，能正确识别这些新 token，不会报错。

👉 所以 tokenizer 会单独保存，不依赖 save_model。
'''
print(f"Model and tokenizer saved to {save_model_dir}")