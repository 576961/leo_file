import os
import re
import time
import torch
from torch import nn

# 从现有模块导入 dataloader, net, vocab_target
from transformer_new import dataloader, net, vocab_target


def find_latest_checkpoint(save_dir='checkpoints'):
    if not os.path.isdir(save_dir):
        return None, 0
    files = os.listdir(save_dir)
    pattern = re.compile(r'transformer_epoch(\d+)\.pt')
    max_epoch = 0
    best = None
    for f in files:
        m = pattern.match(f)
        if m:
            e = int(m.group(1))
            if e > max_epoch:
                max_epoch = e
                best = os.path.join(save_dir, f)
    return best, max_epoch


def resume_train(additional_epochs=5, lr=1e-3, save_dir='checkpoints', max_batches=None, eval_max_batches=2, use_cpu=False):
    device = torch.device('cpu') if use_cpu or not torch.cuda.is_available() else torch.device('cuda')
    model = net.to(device)

    pad_idx = vocab_target.get('<pad>', None)
    vocab_size = len(vocab_target)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_path, last_epoch = find_latest_checkpoint(save_dir)
    if ckpt_path:
        print(f'Loading checkpoint {ckpt_path} (epoch {last_epoch})')
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        try:
            optim.load_state_dict(ckpt.get('optimizer_state_dict', optim.state_dict()))
        except Exception:
            print('Warning: optimizer state could not be loaded (shape mismatch). Continuing with fresh optimizer state.')
    else:
        print('No checkpoint found. Starting from scratch.')

    start_epoch = last_epoch

    for epoch in range(start_epoch + 1, start_epoch + 1 + additional_epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for b_idx, batch in enumerate(dataloader, 1):
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)

            dec_in = tgt[:, :-1]
            labels = tgt[:, 1:]

            optim.zero_grad()
            outputs, _ = model(src, dec_in, None)
            outputs = outputs.reshape(-1, vocab_size)
            labels_flat = labels.reshape(-1)

            loss = loss_fn(outputs, labels_flat)
            loss.backward()
            optim.step()

            total_loss += loss.item()

            if max_batches and b_idx >= max_batches:
                break

        elapsed = time.time() - start_time
        avg_loss = total_loss / (b_idx if b_idx > 0 else 1)
        print(f'Epoch {epoch} | Loss {avg_loss:.4f} | Time {elapsed:.1f}s')

        # 简单评估（快速）
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

                # 将索引序列转换为 token 列表
                for j in range(preds.shape[0]):
                    # 忽略 <bos>，遇到 <eos> 停止
                    ref = []
                    hyp = []
                    for idx in tgt[j, 1:].cpu().tolist():
                        w = next((k for k, v in vocab_target.items() if v == idx), '<unk>')
                        if w == '<eos>':
                            break
                        if w not in ('<pad>', '<bos>'):
                            ref.append(w)
                    for idx in preds[j].cpu().tolist():
                        w = next((k for k, v in vocab_target.items() if v == idx), '<unk>')
                        if w == '<eos>':
                            break
                        if w not in ('<pad>', '<bos>'):
                            hyp.append(w)
                    refs.append(ref)
                    hyps.append(hyp)

                if eval_max_batches and i + 1 >= eval_max_batches:
                    break

        try:
            from nltk.translate.bleu_score import corpus_bleu
            bleu = corpus_bleu([[r] for r in refs], hyps)
            print(f'Eval BLEU (epoch {epoch}): {bleu:.4f}  samples={len(hyps)}')
        except Exception:
            print('BLEU calculation failed (maybe NLTK not available).')

        # 保存检查点
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'transformer_epoch{epoch}.pt')
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict()}, save_path)
        print(f'Saved checkpoint: {save_path}')


if __name__ == '__main__':
    # 默认继续训练 5 个 epoch，可直接编辑本文件顶部的参数来控制
    resume_train(additional_epochs=50, lr=1e-3, save_dir='checkpoints', max_batches=None, eval_max_batches=5, use_cpu=False)
