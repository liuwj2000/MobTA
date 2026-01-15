
import argparse

import sys
import builtins
from datetime import datetime

# ==========================================
# 🕒 【新增功能】自定义 print 函数 (带时间戳 + 强制刷新)
# ==========================================
_original_print = builtins.print

def print(*args, **kwargs):
    """
    自动添加当前时间戳，并且强制 flush，防止日志卡在缓冲区看不到。
    """
    # 获取当前时间
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 强制 flush=True (除非调用时显式设为 False)
    kwargs.setdefault("flush", True)
    
    # 调用原始 print，前面加上时间戳
    _original_print(f"[{now}]", *args, **kwargs)

# 覆盖系统的 print
builtins.print = print
# ==========================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--gpu",
    type=str,
    default="6",
    help="使用的 GPU id，例如 0 或 0,1（默认 0）"
)

parser.add_argument(
    "--mu_L",
    type=float,
    default=0.5,
    help="LoRA 任务向量系数 μ_L（默认 1.0）"
)
parser.add_argument(
    "--mu_E",
    type=float,
    default=0.01,
    help="Embedding 任务向量系数 μ_E（默认 1.0）"
)
parser.add_argument(
    "--mu_H",
    type=float,
    default=0.01,
    help="LM head 任务向量系数 μ_H（默认 1.0）"
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.1,
    help="生成时的 temperature（默认 0.6）"
)

print('start')
args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

#os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"          
# 禁用 torch.compile（排查卡住/结果不对首选）



import gc, torch

import unsloth
from safetensors.torch import load_file
from unsloth import FastLanguageModel, add_new_tokens
from transformers import AutoTokenizer, TextStreamer
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely import wkt 
from transformers import LogitsProcessor
import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed}")

seed_everything(1215)


os.makedirs("../results_qwen", exist_ok=True)
#os.makedirs("../results_gemma", exist_ok=True)
μ_L = args.mu_L
μ_E = args.mu_E
μ_H = args.mu_H
temperature = args.temperature
#out_path = f"../results_qwen/gen_traj_muL{μ_L}_muE{μ_E}_muH{μ_H}_temp{temperature}_seed1215_full.txt"


base_model_name = "unsloth/Llama-3.2-3B"
#base_model_name = "unsloth/Qwen3-1.7B-Base"
#base_model_name ='unsloth/Llama-3.2-1B'#'unsloth/gemma-2-2b'
#base_model_name ='unsloth/Qwen3-1.7B'
#base_model_name ='unsloth/Llama-3.2-3B-Instruct'

out_path = f"../results/{base_model_name.split('/')[1]}_gen_traj_muL{μ_L}_muE{μ_E}_muH{μ_H}_temp{temperature}_seed1215_full.txt"

alpha, r = 64, 32

'''
paths = {
    "A": "../model/outputs/llama_32_3_shanghai_bus_grid_30+interval_new",
    "B": "../model/outputs/llama_32_3_wuxi_bus_grid_30+interval_new",
    "C": "../model/outputs/llama_32_3_wuxi_taxi_grid",
}
'''
paths = {
    "A": "../model/outputs/llama_32_3_shanghai_bus_grid_30+interval_new_mean",
    "B": "../model/outputs/llama_32_3_wuxi_bus_grid_30+interval_new_mean",
    "C": "../model/outputs/llama_32_3_wuxi_taxi_grid_new_mean",
}
'''
paths = {
    "A": "../model/outputs/qwen_shanghai_bus_grid",
    "B": "../model/outputs/qwen_wuxi_bus_grid",
    "C": "../model/outputs/qwen_wuxi_taxi_grid",
}

paths = {
    "A": "../model/outputs/gemma_shanghai_bus_grid",
    "B": "../model/outputs/gemma_wuxi_bus_grid",
    "C": "../model/outputs/gemma_wuxi_taxi_grid",
}

paths = {
    "A": "../model/outputs/qwen_it_shanghai_bus_grid",
    "B": "../model/outputs/qwen_it_wuxi_bus_grid",
    "C": "../model/outputs/qwen_it_wuxi_taxi_grid",
}

paths = {
    "A": "../model/outputs//llama_instruct_shanghai_bus_grid",
    "B": "../model/outputs/llama_instruct_wuxi_bus_grid",
    "C": "../model/outputs/llama_instruct_wuxi_taxi_grid",
}
'''

# =========================================================
# 1️⃣ 加载任务 tokenizer（自动带新增 token）
# =========================================================
print("🔹 Loading model from task model A ...")

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=paths["A"],
        dtype=torch.float32,
        load_in_4bit=False,
        device_map="cuda",
        #fast_inference=False,
)
#print(model)
#print(tokenizer)

############################创建基准E和H的原始值########################
#tok = AutoTokenizer.from_pretrained(paths["A"], use_fast=False)
m, tok = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=512,
        dtype=torch.float32,
        load_in_4bit=False,
        device_map="cpu",
    )
grid=pd.read_csv('../data/shanghai/new_gridize.csv')

new_tokens=grid.loc_id.tolist()

if base_model_name=='unsloth/Qwen3-1.7B-Base' or base_model_name=='unsloth/Qwen3-1.7B':
    m.resize_token_embeddings(len(tok))


add_new_tokens(m, tok, new_tokens=new_tokens, method="mean")

E_ref = m.model.embed_tokens.weight.detach().cpu().float().clone()
H_ref = m.lm_head.weight.detach().cpu().float().clone()
del m
gc.collect()

'''
new_token_ids = torch.tensor(
    [tokenizer.convert_tokens_to_ids(t) for t in new_tokens],
    dtype=torch.long
)
new_token_ids = new_token_ids[new_token_ids >= 0].unique()
hash_id = tokenizer.encode("#", add_special_tokens=False)[0]
eos_id  = tokenizer.eos_token_id
safe_ids = torch.tensor([hash_id, eos_id], dtype=torch.long)

ids_to_apply = torch.unique(torch.cat([new_token_ids, safe_ids]))
'''
#################################################################


# 确保 embedding 尺寸与 tokenizer 一致
if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
    model.resize_token_embeddings(len(tokenizer))
    print(f"✅ resized embeddings to {len(tokenizer)}")


# =========================================================
# 3️⃣ 加载三个 LoRA adapter
# =========================================================
print("🔹 Loading LoRA adapters ...")

L_B = load_file(f"{paths['B']}/adapter_model.safetensors")
L_C = load_file(f"{paths['C']}/adapter_model.safetensors")


# =========================================================
# 4️⃣ 融合 LoRA 参数 (A+B−C)
# =========================================================
print("🔹 Merging LoRA into full weights ...")
for name, module in model.named_modules():
    if hasattr(module, "lora_A"):
        key_A = f"{name}.lora_A.weight"
        key_B = f"{name}.lora_B.weight"
        if key_A not in L_B:
            print('WARNING!  no lora: ',key_A)
            continue

        if hasattr(module, "base_layer"):
            target_layer = module.base_layer
        else:
            # 防御性代码：如果某些层没有被封装 (虽然看你的结构应该都有)
            target_layer = module
        print(target_layer)

        w_a_B = L_B[key_A].to(torch.float32).to(model.device)
        w_b_B = L_B[key_B].to(torch.float32).to(model.device)
        
        w_a_C = L_C[key_A].to(torch.float32).to(model.device)
        w_b_C = L_C[key_B].to(torch.float32).to(model.device)

        # 计算 B 和 C 的 update matrix (ΔW = B @ A)
        # 形状通常是 [out_features, in_features]
        delta_B = w_b_B @ w_a_B
        delta_C = w_b_C @ w_a_C
        
        # 计算任务向量: (B - C)
        task_vector = delta_B - delta_C
        
        # 应用你的公式: - μ_L * (B - C) * scaling
        # 这意味着你在减去 (B-C) 的特征
        injection = -1 * μ_L * task_vector *(alpha / r)

        # 5. 注入到 Base Layer
        with torch.no_grad():
            # 确保设备和类型一致
            injection = injection.to(target_layer.weight.device, dtype=target_layer.weight.dtype)
            target_layer.weight.add_(injection)
            
        print(f"✅ Merged into: {name}")

        # 内存清理 (非常重要，否则显存会爆)
        del w_a_B, w_b_B, w_a_C, w_b_C, delta_B, delta_C, task_vector, injection
        
    # 定期清空 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

gc.collect()
print("🎉 Injection done.")


# =========================================================
# 5️⃣ 计算 ΔE/ΔH 并融合 (embedding + lm_head)
# =========================================================
'''
def get_delta_EH(path):
    W = load_file(f"{path}/adapter_model.safetensors")

    E_train = W["base_model.model.model.embed_tokens.weight"].cpu().float()
    H_train = W["base_model.model.lm_head.weight"].cpu().float()

    
    ΔE = E_train - E_ref
    ΔH = H_train - H_ref
    return ΔE, ΔH
    #对每个任务 (A/B/C) 都算出自己的 embedding / head 任务向量
'''
def get_delta_EH(path):
    m, _ = FastLanguageModel.from_pretrained(
        model_name=path,
        dtype=torch.float32,
        load_in_4bit=False,
        device_map="cpu",
        #fast_inference=False,
    )



    #E_base  = m.model.model.embed_tokens.original_module.weight.detach().cpu()
    E_train = m.model.model.embed_tokens.modules_to_save["default"].weight.detach().cpu().float()
    #H_base  = m.model.lm_head.original_module.weight.detach().cpu()
    H_train = m.model.lm_head.modules_to_save["default"].weight.detach().cpu().float()
    ΔE = E_train - E_ref
    ΔH = H_train - H_ref
    del m; 
    gc.collect()
    return ΔE, ΔH
    #对每个任务 (A/B/C) 都算出自己的 embedding / head 任务向量

print("🔹 Computing ΔE / ΔH task vectors ...")
ΔE_B, ΔH_B = get_delta_EH(paths["B"])
ΔE_C, ΔH_C = get_delta_EH(paths["C"])


E_A_train = model.base_model.model.model.embed_tokens.modules_to_save["default"].weight
H_A_train = model.base_model.model.lm_head.modules_to_save["default"].weight


inj_E = -μ_E * (ΔE_B - ΔE_C)
inj_H = -μ_H * (ΔH_B - ΔH_C)
#inj_E=inj_E.to(E_A_train.device)
#inj_H=inj_H.to(E_A_train.device)
with torch.no_grad():
    #ids = ids_to_apply.to(E_A_train.device)
    # dtype/device 对齐
    #E_A_train[ids].add_(inj_E[ids].to(device=E_A_train.device, dtype=E_A_train.dtype))
    #H_A_train[ids].add_(inj_H[ids].to(device=H_A_train.device, dtype=H_A_train.dtype))
    
    
    E_A_train.add_(inj_E.to(device=E_A_train.device, dtype=E_A_train.dtype))
    H_A_train.add_(inj_H.to(device=H_A_train.device, dtype=H_A_train.dtype))
#至此：LoRA + embedding + lm_head 三个部分都完成 A−(B−C) 融合，得到一个新的“任务向量模型”。



FastLanguageModel.for_inference(model)


start_loc_probs = np.load("../data/shanghai/shanghai_bus_start.npy", allow_pickle=True)
start_time_probs = np.load("../data/shanghai/shanghai_bus_starttime_min_new.npy", allow_pickle=True)
traj_len_probs= np.load("../data/shanghai/shanghai_bus_trajlen.npy", allow_pickle=True)
len(start_loc_probs),len(start_time_probs),len(traj_len_probs)
#之前统计过起点 grid / 起始分钟 / 轨迹长度的经验分布，现在用来作为条件分布去采样。

loc_ids = np.arange(len(start_loc_probs))      # 位置索引
time_vals = np.arange(len(start_time_probs))   # 时间索引（分钟）
len_vals = np.arange(1, len(traj_len_probs)+1) # 轨迹长度值（从1开始更自然）

# ✅ 示例采样一条
start_loc = int(np.random.choice(loc_ids, p=start_loc_probs))
start_time = int(np.random.choice(time_vals, p=start_time_probs))
target_len = int(np.random.choice(len_vals, p=traj_len_probs))













################# 添加logit processor，loc 只选择二阶邻居的location #####################

grid_gdf_sh = pd.read_csv('../data/shanghai/new_gridize.csv')
# 1️⃣ 把 geometry 这一列从字符串变成真正的 Polygon
grid_gdf_sh['geometry'] = grid_gdf_sh['geometry'].apply(wkt.loads)
grid_gdf_sh = gpd.GeoDataFrame(grid_gdf_sh, geometry='geometry', crs="EPSG:4326")


# 用 sjoin 找 touches 的格子
joined = gpd.sjoin(grid_gdf_sh, grid_gdf_sh, predicate='touches', how='inner') 
'''
对于网格（grid），如果两个格子的 polygon 有边或点相接，就可以认为它们是邻居。
找所有 geometry A touches geometry B 的组合

grid_gdf_sh 左表：所有格子

grid_gdf_sh 右表：也是所有格子

predicate='touches'：只保留左格子 touches 右格子的情况

结果 joined 会是一个 笛卡尔匹配之后过滤 touches 的表
'''


neighbors_1 = {
    int(row['spiral_index']): set()
    for _, row in grid_gdf_sh.iterrows()
}
for _, row in joined.iterrows(): 
    i = int(row['spiral_index_left']) 
    j = int(row['spiral_index_right']) 
    neighbors_1[i].add(j)
#把邻居关系写入邻接表 neighbors_1
neighbors_1
'''
{0: {1, 2, 3, 4, 5, 6, 7, 8},
 1: {0, 2, 3, 7, 8, 9, 10, 24},
 2: {0, 1, 3, 9, 10, 11, 12, 13},
 ...
'''


neighbors_2 = {}
#二阶邻居 = 自己 + 一阶邻居 + 一阶邻居的一阶邻居
for i, nbs in neighbors_1.items():
    second = set(nbs) | {i}
    #先把“一阶邻居 + 自己”放进去
    for j in nbs:
        #这个循环遍历 i 的每个一阶邻居 j
        second |= neighbors_1.get(j, set())
        #neighbors_1[j] 是格子 j 的一阶邻居集合
        #把「j 的邻居」并入 second
    neighbors_2[i] = second

'''
写一个自定义 LogitsProcessor 来限制下一个位置 token

在每一步生成时，根据上一个位置 token，把不在它二阶邻居中的 token 的 logits 设为 -inf，这样模型就只能在允许的 里采样。
'''

# 1) loc token 文本到 token_id 的映射
loc_token_to_id = {}
for loc_id in range(0, 1056):  # 根据你真实的最大 grid id 调整
    tok = f"<LOC_{loc_id}>"
    if tok in tokenizer.get_vocab():
        loc_token_to_id[loc_id] = tokenizer.convert_tokens_to_ids(tok)

loc_token_to_id
'''
{0: 128256,
 1: 128257,
 2: 128258,
 ...
'''

loc_token_id_set = set(loc_token_to_id.values())

#把 neighbors_2 换成 “token_id -> 允许的 token_id 集合”：
neighbor2_token = {}
for loc_id, nb_set in neighbors_2.items():
    if loc_id not in loc_token_to_id:
        continue
    this_tok_id = loc_token_to_id[loc_id]
    allowed_tok_ids = []
    for nb in nb_set:
        if nb in loc_token_to_id:
            allowed_tok_ids.append(loc_token_to_id[nb])
    neighbor2_token[this_tok_id] = set(allowed_tok_ids)

import re
'''
只有当序列末尾出现 “location is” 时，模型才进入位置生成模式
➤ 进入位置生成模式后：

所有非 <LOC_*> 的 token logits 被置为 -∞（强制下一 token 必须是 <LOC_x>）

所有不在二阶邻居中的 <LOC_*> logits 也被置为 -∞（强制只能在邻域内生成）
'''

class LocationAwareSecondOrderNeighborProcessor(LogitsProcessor):
    def __init__(
        self,
        neighbor2_token: dict, #二阶邻居的 token 级别映射：
        loc_token_id_set: set, #所有 <LOC_x> token 的 id 集合,用于判断一个 token 是否属于位置类别
        tokenizer, #需要 decode 最近一段 token 来判断是否进入 “location 模式”
        loc_trigger_pattern: str = r"location is",  # 触发“下一步是位置”的模式
        max_check_len: int = 20,#decode 多少个 token 用来匹配 trigger（避免 decode 太长）
        penalty: float = -1e10, #用于 mask 的值，一般设为 -1e10（近似 -∞）
    ):
        super().__init__()
        self.neighbor2_token = neighbor2_token      # dict: last_loc_token_id -> {allowed_loc_token_ids}
        '''
         key：某个位置 token 的 token_id（比如 <LOC_19> 对应的 id）

        value：一个 set，里面是 允许出现在下一个时间步的 <LOC_*> 的 token_id 集合
        '''
        self.loc_token_id_set = loc_token_id_set    # 所有 <LOC_*> 的 token_id
        self.tokenizer = tokenizer
        self.max_check_len = max_check_len
        self.penalty = penalty

        # 检测末尾是否是 "... location is [若干空格]" 的正则
        self.trigger_re = re.compile(loc_trigger_pattern + r"\s*$", re.IGNORECASE)
        '''
        匹配文本以 "location is" 结尾

        忽略大小写
        '''

    def _should_constrain_locations(self, seq: torch.LongTensor) -> bool:
        """
        判断当前这一步：下一 token 是否应该是 <LOC_x>？
        做法：decode 序列最后 max_check_len 个 token，看文本结尾是不是 'location is'
        """
        # 截取最后一小段，避免 decode 太长
        sub_seq = seq[-self.max_check_len:].tolist()
        
        text = self.tokenizer.decode(sub_seq, skip_special_tokens=True)
        #最后一小段 decode 成文本
        
        text = text.lower()
        # 例如 "... arrival time is 1116, location is"
        return bool(self.trigger_re.search(text))
        #用正则检查是否以 “location is” 结尾
        '''
        如果匹配，则告诉 processor：

        🔥 现在应该生成 <LOC_x> 了！

        因此你就进入下一步“强制位置 token 阶段”。
        '''

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """
        是 HF 在每个 decoding step 自动调用的方法。
        input_ids: [batch, seq_len] 当前已生成序列
        scores:    [batch, vocab_size]  （未 softmax 的 logits） 下一 token 的 logits
        """
        bsz, seq_len = input_ids.shape
        vocab_size = scores.size(-1)

        # 备份原始 logits，后面恢复用
        original_scores = scores.clone()

        # 提前准备一个所有 id 的 tensor（方便全量置 penalty）
        # 也可以直接 logits[:] = penalty，不用这个
        for b in range(bsz):
            seq = input_ids[b]

            # 1️⃣ 判断现在是不是“该生成 <LOC_x> 的时刻”
            if not self._should_constrain_locations(seq):
                # 不是生成位置的环节：完全不动 logits
                continue
            '''
            如果现在不是 "location is" 之后：

            📌 什么也不做

            让模型自由生成（比如 arrival time 的数字）。
            '''

            # 2️⃣ 找“最近的一个位置 token”（当前步的下一位置，要基于上一个位置的邻域）
            last_loc_token_id = None
            for pos in range(seq_len - 1, -1, -1):
                tok_id = seq[pos].item()
                if tok_id in self.loc_token_id_set:
                    last_loc_token_id = tok_id
                    break
            #找到最近的一个 <LOC_x> token

            # 如果还没出现过位置 token，或者不在邻接表里，就不约束
            if last_loc_token_id is None:
                continue

            allowed = self.neighbor2_token.get(last_loc_token_id, None)
            #取该 token 的二阶邻居集合
            if not allowed:
                # 这个位置没有邻居信息，就不强行限制
                continue

            # 3️⃣ 开始真正改 logits
            logits = scores[b]
            orig_logits = original_scores[b]

            # 3.1 先把所有 token 都打成 penalty：下一步只会在“允许的位置 token”里选
            logits.fill_(self.penalty)

            # 3.2 只恢复“允许的二阶邻居位置 token”的原始 logits
            allowed_tensor = torch.tensor(list(allowed), device=logits.device, dtype=torch.long)
            logits[allowed_tensor] = orig_logits[allowed_tensor]

            # ✅ 此时：
            #  - 非 <LOC_*> token：logits = penalty
            #  - 不在二阶邻居中的 <LOC_*>：logits = penalty
            #  - 在二阶邻居中的 <LOC_*>：logits = 原来的 logits

            scores[b] = logits

        return scores

neighbor_processor = LocationAwareSecondOrderNeighborProcessor(
    neighbor2_token=neighbor2_token,
    loc_token_id_set=loc_token_id_set,
    tokenizer=tokenizer,
    loc_trigger_pattern=r"location is ",  # 如果你的 prompt 写法固定，就用这个
    max_check_len=20,
    penalty=-1e10,
)



#############################################################################################
















eos_token = tokenizer.eos_token



def generate_single_trajectory(start_loc, start_time, target_len, temperature,max_retries=5):
    """生成单条轨迹，带 eos 截断、长度限制与格式检查。"""
    if base_model_name=='unsloth/Qwen3-1.7B-Base':
        prompt = f"arrival time is {start_time}, location is <LOC_{start_loc}> # "
    elif base_model_name=='unsloth/gemma-2-2b':
        prompt = f"arrival time is {start_time}, location is <LOC_{start_loc}> # "
    else:
        prompt = f"arrival time is {start_time}, location is <LOC_{start_loc}> #"
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    #给模型的 prompt 就是轨迹的第一对 (time, loc)，后面用 # 作为一对一对的分隔符，让模型续写下一对。
    #print('#'*100)
    #print(prompt)
    for attempt in range(max_retries):
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=int(min(target_len * 15, 500)),  # 给足空间
                temperature=temperature,
                top_k=10,
                #logits_processor=[neighbor_processor],
            )

        text = tokenizer.decode(output[0], skip_special_tokens=False)
        #print(text)
        # === 1️⃣ 截断到 <end_of_text> ===
        if eos_token in text:
            text = text.split(eos_token)[0] + eos_token

        # === 2️⃣ 截取指定数量的时间-位置对 ===
        segments = text.split("#")
        if len(segments) - 1 > target_len:
            text = "#".join(segments[:target_len]) + "#"
        #len(segments) - 1 是当前 text 里有多少个 #，大致等于 pairs 数量。太多就截断。

        # 2️⃣.5 按最后合法对截断（无论有没有 #）
        pattern_pair = r"arrival time is \d+, location is <LOC_\d+>"
        matches_iter = list(re.finditer(pattern_pair, text))
        if len(matches_iter) > 0:
            last_end = matches_iter[-1].end()
            text = text[:last_end] + eos_token

        # === 3️⃣ 检查格式正确性 ===
        # 格式要求：必须至少有 target_len 个 "arrival time is X, location is <LOC_Y>"
        pattern = r"arrival time is \d+, location is <LOC_\d+>"
        matches = re.findall(pattern, text)
        if len(matches) >= min(5, target_len):  # 至少有几对才算合格
            #至少要匹配到 min(5, target_len) 对 (time,loc) 才算合格，不然就重试。
            return text  # ✅ 合格直接返回
        else:
            
            print(f"⚠️ 第 {attempt+1} 次生成格式错误，重试中...")
            print(text)
            continue  # 再试一次

    print("❌ 达到最大重试次数，放弃该样本")
    return None  # 全部失败则舍弃


import re
num_samples=500#2000#67143
i=0
generated_trajs = []
while len(generated_trajs) < num_samples:
    start_loc = int(np.random.choice(loc_ids, p=start_loc_probs))
    start_time = int(np.random.choice(time_vals, p=start_time_probs))
    target_len = int(np.random.choice(len_vals, p=traj_len_probs))

    text = generate_single_trajectory(start_loc, start_time, target_len,temperature)

    if text is not None:  # ✅ 格式正确才计入
        generated_trajs.append(text)
        i += 1

        if i % 10 == 0:
            print(f"✅ 已生成 {i}/{num_samples} 条")
    else:
        continue
    
with open(out_path, "w", encoding="utf-8") as f:
    for line in generated_trajs:
        # 保证每条一行，末尾换行
        f.write(line.strip() + "\n")

print(f"✅ Saved {len(generated_trajs)} trajectories to {out_path}")

import re
traj_real_time = []  # 存每条轨迹的时间序列
traj_real_loc = []   # 存每条轨迹的地点序列
with open('../data_processing/real_traj_shanghai_taxi.txt') as f:
    for line in f:
        line = line.strip()
        if not line:
                continue
        # 去掉 <end_of_text>，避免干扰正则
        line = line.replace("<end_of_text>", "")

        # 正则匹配所有时间和位置对
        pairs = re.findall(r"arrival time is (\d+), location is (<LOC_\d+>)", line)

        if len(pairs) == 0:
            continue  # 跳过空轨迹

        # 分别提取时间和位置
        times = [int(t) for t, _ in pairs]
        locs = [loc for _, loc in pairs]

        traj_real_time.append(times)
        traj_real_loc.append(locs)

import re
gen_real_time = []  # 存每条轨迹的时间序列
gen_real_loc = []   # 存每条轨迹的地点序列
for line in generated_trajs:
        line = line.strip()
        if not line:
                continue
        # 去掉 <end_of_text>，避免干扰正则
        line = line.replace(eos_token, "")

        # 正则匹配所有时间和位置对
        pairs = re.findall(r"arrival time is (\d+), location is (<LOC_\d+>)", line)

        if len(pairs) == 0:
            continue  # 跳过空轨迹

        # 分别提取时间和位置
        times = [int(t) for t, _ in pairs]
        locs = [loc for _, loc in pairs]

        gen_real_time.append(times)
        gen_real_loc.append(locs)

import pandas as pd
grid_df = pd.read_csv('../data/shanghai/new_gridize.csv')
from shapely import wkt
# 2️⃣ 将 POLYGON 转为 shapely 对象
grid_df['geometry'] = grid_df['geometry'].apply(wkt.loads)

# 3️⃣ 提取每个 polygon 的中心点经纬度
grid_df['lon'] = grid_df['geometry'].apply(lambda x: x.centroid.x)
grid_df['lat'] = grid_df['geometry'].apply(lambda x: x.centroid.y)


import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from math import radians, sin, cos, sqrt, atan2

# ============================================================
# 一、准备：定义工具函数
# ============================================================

# 经纬度距离（单位：km）
def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

# 提取 <LOC_xxx> 的 id
def extract_id(loc_str):
    return int(loc_str.strip('<>').split('_')[1])

# 将 loc 转换为 (lon, lat)
loc2coord = dict(zip(grid_df['loc_id'], zip(grid_df['lon'], grid_df['lat'])))
def loc_to_coord(loc):
    return np.array(loc2coord[loc])

# 轨迹总距离
'''
遍历轨迹中的每一个点，使用 loc_to_coord 获取坐标
然后用 haversine 函数累加计算每两个连续点之间的距离。
'''
def compute_distance(traj):
    coords = [loc_to_coord(loc) for loc in traj]
    return sum(haversine(*coords[i-1], *coords[i]) for i in range(1, len(coords)))

# 轨迹半径 gyration
'''
首先计算该轨迹所有点的“质心”（即坐标的平均值）
然后计算所有点到这个质心的平均距离（技术上是均方根距离）
'''
def compute_radius(traj):
    coords = np.array([loc_to_coord(loc) for loc in traj])
    if len(coords) < 2:
        return 0
    center = coords.mean(axis=0)
    return np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))

# 每日访问地点数
'''
计算（通常是一天内的）访问过的不同地点的数量。
'''
def compute_daily_loc(traj):
    return len(set(traj))

# G-rank（每个位置访问频率）
'''
计算地点的访问频率分布
'''
def compute_g_rank(traj):
    counts = pd.Series(traj).value_counts()
    return counts / counts.sum()

def compute_duration_with_time(times, locs):
    """
    根据时间序列和位置序列计算连续停留时长（分钟）
    times: [t1, t2, t3, ...]   每个时间点（单位：分钟）
    locs:  [<LOC_x>, <LOC_y>, ...] 对应的位置
    返回: 每个停留段的时长（分钟）
    """
    durations = []
    start_time = times[0]
    current_loc = locs[0]

    for i in range(1, len(locs)):
        if locs[i] != current_loc:
            # 出现位置变化 → 计算上一段的停留时长
            dur = times[i-1] - start_time
            durations.append(max(dur, 0))  # 防止异常负数
            # 开始新段
            start_time = times[i]
            current_loc = locs[i]

    # 最后一段也要计算
    durations.append(times[-1] - start_time)
    return durations


from tqdm import tqdm
def compute_metrics_for_all(traj_list,time_list):
    metrics = {
        "distance": [],
        "radius": [],
        "duration_all": [],  # 展开所有停留时长
        "daily_loc": [],
        "g_rank_list": []    # 每条轨迹的访问频率分布
    }
    
    for i in tqdm(range(len(traj_list))):
        traj=traj_list[i]
        time=time_list[i]
        if len(traj) < 2:
            continue  # 跳过长度过短的轨迹
        metrics["distance"].append(compute_distance(traj))
        metrics["radius"].append(compute_radius(traj))
        '''
        遍历 traj_list 中的每一条轨迹 traj。

        计算该轨迹的 distance, radius, daily_loc，并将结果 append（添加）到 metrics 字典中对应的列表。
        '''
        metrics["duration_all"].extend(compute_duration_with_time(time,traj))
        '''
        计算 duration，并使用 .extend() 将其展平。
        例如，两条轨迹的停留时长分别是 [3, 1] 和 [2, 5]，duration_all 会变成 [3, 1, 2, 5]。
        这是因为我们关心的是所有停留时长的总体分布，而不是每条轨迹的停留时长。
        '''
        metrics["daily_loc"].append(compute_daily_loc(traj))
        metrics["g_rank_list"].append(compute_g_rank(traj))
        '''
        计算 g_rank，并将这个 pd.Series 对象添加到 g_rank_list 中
        '''

    return metrics

real_metrics = compute_metrics_for_all(traj_real_loc,traj_real_time)
gen_metrics = compute_metrics_for_all(gen_real_loc,gen_real_time)



# ============================================================
# 三、定义 JSD 计算函数
# ============================================================

def jsd_continuous(real_vals, gen_vals, bins=30):
    """计算连续变量分布的JSD"""
    real_hist, bin_edges = np.histogram(real_vals, bins=bins, density=True)
    gen_hist, _ = np.histogram(gen_vals, bins=bin_edges, density=True)
    real_hist += 1e-12
    gen_hist += 1e-12
    return jensenshannon(real_hist, gen_hist)**2

# ============================================================
# 四、计算每个指标的JSD
# ============================================================

# 1️⃣ Distance
JSD_distance = jsd_continuous(real_metrics["distance"], gen_metrics["distance"])

# 2️⃣ Radius of gyration
JSD_radius = jsd_continuous(real_metrics["radius"], gen_metrics["radius"])

# 3️⃣ Duration
JSD_duration = jsd_continuous(real_metrics["duration_all"], gen_metrics["duration_all"])

# 4️⃣ DailyLoc
JSD_daily = jsd_continuous(real_metrics["daily_loc"], gen_metrics["daily_loc"])


# 5️⃣ G-rank（合并所有轨迹的频率）
all_locs = list(grid_df['loc_id'])
def aggregate_g_rank(g_list):
    total = sum(g.reindex(all_locs, fill_value=0) for g in tqdm(g_list))
    return total / total.sum()
g_real = aggregate_g_rank(real_metrics["g_rank_list"])
g_gen = aggregate_g_rank(gen_metrics["g_rank_list"])
JSD_g_rank = jensenshannon(g_real, g_gen)**2



print("🔹 Jensen–Shannon散度（越小越好）")
print(f"Distance: {JSD_distance:.4f}")
print(f"Radius:   {JSD_radius:.4f}")
print(f"Duration: {JSD_duration:.4f}")
print(f"DailyLoc: {JSD_daily:.4f}")
print(f"G-rank:   {JSD_g_rank:.4f}")


