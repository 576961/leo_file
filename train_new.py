
import os
from sched import scheduler
from sched import scheduler
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import time
from transformer_new import dataloader, net, vocab_target, tokenizer, source_tensor, target_tensor
from nltk.translate.bleu_score import corpus_bleu
import re
import glob

model = net
vocab = vocab_target
train_dataloader = dataloader

# 假设 vocab 包含特殊标记索引：pad_idx, bos_idx, eos_idx
pad_idx = vocab['[PAD]']
bos_idx = vocab['[CLS]']
eos_idx = vocab['[SEP]']


# 初始化时设置 device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
global_step = 0
# 训练配置
accum_steps = 8                # 梯度累积步数
effective_batch_size = 128 * accum_steps  # 有效批量大小（句子对）
total_updates = 100000         # 原论文基础模型的更新步数（10万步）
total_batches = total_updates * accum_steps  # 总小批量数

warmup_updates = 4000          # warmup 步数（按参数更新计）
d_model = 512                  # 模型维度，用于学习率计算

# 优化器：使用论文的 Adam 参数
# 设置基础学习率为 1.0，使调度器产生论文中期望的学习率曲线
optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

# 学习率调度函数（按参数更新步数计算）
def lr_lambda(step):
    # step: 当前参数更新次数（从1开始）
    if step == 0:
        step = 1  # 避免除以0
    # 论文公式：lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})
    warmup_steps = warmup_updates
    scale = d_model ** (-0.5)
    arg1 = step ** (-0.5)
    arg2 = step * (warmup_steps ** (-1.5))
    return scale * min(arg1, arg2)

# 使用 LambdaLR，注意 step 对应 optimizer 更新次数
scheduler = LambdaLR(optimizer, lr_lambda)

# 训练循环
model.train()
optimizer.zero_grad()
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx) #maskedCEL, 忽略填充编码
os.makedirs("checkpoints_new", exist_ok=True)
ckpt_dir = "checkpoints_new"
def find_latest_checkpoint(directory):
    pattern = os.path.join(directory, "transformer_*.pth")
    files = glob.glob(pattern)
    if not files:
        return None
    # try to extract step/epoch number from filename, fallback to mtime
    def key(f):
        m = re.search(r"(step|epoch)_(\d+)", f)
        if m:
            return int(m.group(2))
        return int(os.path.getmtime(f))
    latest = max(files, key=key)
    return latest

# 尝试从已有检查点恢复训练状态（model + optimizer + scheduler + global_step）
latest_ckpt = find_latest_checkpoint(ckpt_dir)
if latest_ckpt is not None:
    try:
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        if 'optimizer_state' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state'])
                # 将 optimizer 的 tensor 移到当前 device
                for state in optimizer.state.values():
                    for k, v in list(state.items()):
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            except Exception:
                print(f"Warning: failed to fully load optimizer state from {latest_ckpt}")
        if 'scheduler_state' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state'])
            except Exception:
                print(f"Warning: failed to load scheduler state from {latest_ckpt}")
        global_step = int(ckpt['global_step'])
        print(f"Loaded checkpoint {latest_ckpt} (global_step={global_step})")

    except Exception as e:
        print(f"Failed to load checkpoint {latest_ckpt}: {e}")

last_log_time = time.time()
def compute_bleu_batch(model, tokenizer, source_tensor, target_tensor, device, num_samples=256):
    model.eval()
    with torch.no_grad():
        n = min(num_samples, source_tensor.size(0))
        idx = torch.arange(n)
        src = source_tensor[idx].to(device)
        max_len = target_tensor.size(1)
        batch = src.size(0)
        dec_input = torch.full((batch, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            outputs, _ = model(src, dec_input, None)
            next_logits = outputs[:, -1, :]
            next_tok = next_logits.argmax(dim=-1, keepdim=True)
            dec_input = torch.cat([dec_input, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(1) == eos_idx)
            if finished.all():
                break

        hyps = []
        refs = []
        for i in range(batch):
            hyp_ids = dec_input[i].cpu().tolist()
            hyp_str = tokenizer.decode(hyp_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            ref_ids = target_tensor[idx[i]].cpu().tolist()
            ref_str = tokenizer.decode(ref_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            hyps.append(hyp_str.split())
            refs.append([ref_str.split()])

        bleu = corpus_bleu(refs, hyps)
    model.train()
    return bleu

def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])

model.apply(xavier_init_weights)

flag = True
while flag:
    for batch_idx, batch in enumerate(train_dataloader):
        src, src_valid_lens, tgt, tgt_valid_lens = batch
        src = src.to(device)
        src_valid_lens = src_valid_lens.to(device)
        tgt = tgt.to(device)
        tgt_valid_lens = tgt_valid_lens.to(device)

        # decoder 输入为去掉最后一个 token 的目标，标签为去掉第一个 token 的目标
        dec_in = tgt[:, :-1]
        labels = tgt[:, 1:]

        outputs, _ = model(src, dec_in, src_valid_lens)  # outputs.shape = (batch, seq-1, vocab_size)
        outputs = outputs.reshape(-1, len(vocab))
        labels_flat = labels.reshape(-1)

        loss = loss_fn(outputs, labels_flat)

        # 将损失除以累积步数，使累积后的梯度平均
        loss = loss / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1


            # 打印日志（每一定更新次数）
            if global_step % 2 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                # 计算自上一次打印日志经过的时间
                now = time.time()
                elapsed = now - last_log_time
                last_log_time = now
                # 使用科学计数法打印学习率，避免非常小的 lr 被显示为 0.000000
                print(f"Update step {global_step}, Loss: {loss.item() * accum_steps:.4f}, LR: {current_lr:.8e}, elapsed: {elapsed:.3f}s")

                with open("loss.txt", 'a', encoding='utf-8') as f:
                    f.write(f"{global_step} {loss.item() * accum_steps:.4f}\n")

            #每100个global_step保存一次模型
            '''if global_step % 2 == 0:
                path = os.path.join(ckpt_dir, f"transformer_step_{global_step}.pth")
                to_save = {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'global_step': global_step
                }
                try:
                    torch.save(to_save, path)
                except Exception as e:
                    # 回退保存仅模型权重
                    print(f"Warning: failed to save full checkpoint: {e}; falling back to model.state_dict()")
                    torch.save(model.state_dict(), path)
                
            # 每20个更新步数评估一次 BLEU 分数（使用前256个样本）    
            if global_step % 20 == 0:
                try:
                    bleu_score = compute_bleu_batch(model, tokenizer, source_tensor, target_tensor, device, num_samples=256)
                    print(f"BLEU (first 256 samples) at step {global_step}: {bleu_score:.4f}")
                except Exception as e:
                    print(f"BLEU eval failed at step {global_step}: {e}")'''


            # 达到总更新次数时停止
            if global_step >= total_updates:
                print("Training finished.")
                flag = False
                break

