import os
import time
import torch
from torch import nn
from nltk.translate.bleu_score import corpus_bleu

# 从现有模块导入 dataloader, net, vocab_target
from transformer_new import dataloader, net, vocab_target


# 简单的配置字典，直接修改这些值来控制训练
CONFIG = {
    'epochs': 5,
    'lr': 1e-3,
    'save_dir': 'checkpoints_2',
    'eval_every': 1,
    'max_batches': 50,        # None 表示不限制批次数
    'eval_max_batches': 5,      # 评估时最多使用的批次数，用于快速计算 BLEU
    'use_cpu': False,
}


def build_idx2word(vocab):
    return {idx: word for word, idx in vocab.items()}


def seq_to_tokens(seq, idx2word):
    """把索引序列转换为单词列表，遇到 <eos> 停止，忽略 <pad>/<bos>。"""
    out = []
    for i in seq:
        w = idx2word.get(int(i), '<unk>')
        if w == '<eos>':
            break
        if w not in ('<pad>', '<bos>'):
            out.append(w)
    return out


def compute_bleu(refs, hyps):
    # corpus_bleu 需要每个参考是一个参考列表
    return corpus_bleu([[r] for r in refs], hyps)


def train_simple(cfg):
    device = torch.device('cpu') if cfg['use_cpu'] or not torch.cuda.is_available() else torch.device('cuda')
    model = net.to(device)

    pad_idx = vocab_target.get('<pad>', None)
    vocab_size = len(vocab_target)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    idx2word = build_idx2word(vocab_target)
    os.makedirs(cfg['save_dir'], exist_ok=True)

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss = 0.0
        start = time.time()

        for b_idx, batch in enumerate(dataloader, 1):
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)

            # decoder 输入为去掉最后一个 token 的目标，标签为去掉第一个 token 的目标
            dec_in = tgt[:, :-1]
            labels = tgt[:, 1:]

            optim.zero_grad()
            outputs, _ = model(src, dec_in, None)  # (batch, seq-1, vocab_size)
            outputs = outputs.reshape(-1, vocab_size)
            labels_flat = labels.reshape(-1)

            loss = loss_fn(outputs, labels_flat)
            loss.backward()
            optim.step()

            total_loss += loss.item()

            if cfg['max_batches'] and b_idx >= cfg['max_batches']:
                break

        elapsed = time.time() - start
        avg_loss = total_loss / (b_idx if b_idx > 0 else 1)
        print(f"Epoch {epoch} | Loss {avg_loss:.4f} | Time {elapsed:.1f}s")

        # 评估并保存模型
        if epoch % cfg['eval_every'] == 0:
            model.eval()
            refs, hyps = [], []
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    src, tgt = batch
                    src = src.to(device)
                    tgt = tgt.to(device)

                    dec_in = tgt[:, :-1]
                    outputs, _ = model(src, dec_in, None)
                    preds = outputs.argmax(dim=-1)

                    for j in range(preds.shape[0]):
                        ref = seq_to_tokens(tgt[j, 1:].cpu().tolist(), idx2word)
                        hyp = seq_to_tokens(preds[j].cpu().tolist(), idx2word)
                        refs.append(ref)
                        hyps.append(hyp)

                    if cfg['eval_max_batches'] and i + 1 >= cfg['eval_max_batches']:
                        break

            bleu = compute_bleu(refs, hyps)
            print(f"Eval BLEU (epoch {epoch}): {bleu:.4f}  samples={len(hyps)}")

            # 保存模型参数
            path = os.path.join(cfg['save_dir'], f"transformer_epoch{epoch}.pt")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optim_state_dict': optim.state_dict()}, path)
            print(f"Saved: {path}")


if __name__ == '__main__':
    # 直接用 CONFIG 控制训练，避免 argparse 的复杂性
    train_simple(CONFIG)

