import os
import re
import argparse
import torch
from nltk.translate.bleu_score import corpus_bleu

from transformer_new import dataloader, net, vocab_target, vocab_source


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


def idx2word_map(vocab):
    return {idx: w for w, idx in vocab.items()}


def greedy_decode(model, src_batch, max_len, bos_idx, eos_idx, device):
    # src_batch: (batch, src_len)
    encoder = model.encoder
    decoder = model.decoder

    enc_outputs = encoder(src_batch, None)
    state = decoder.init_state(enc_outputs, None)

    batch_size = src_batch.shape[0]
    generated = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)

    # Greedy decode kept for compatibility; prefer beam_search_decode below.
    for _ in range(max_len):
        outputs, state = decoder(generated, state)
        next_logits = outputs[:, -1, :]
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if ((next_token == eos_idx).all()):
            break
    return generated[:, 1:]


def beam_search_decode(model, src_batch, max_len, bos_idx, eos_idx, device, beam_size=5):
    # Implements per-sample beam search (batch dimension processed one-by-one)
    encoder = model.encoder
    decoder = model.decoder

    results = []
    batch_size = src_batch.shape[0]
    for i in range(batch_size):
        src = src_batch[i].unsqueeze(0)
        enc_outputs = encoder(src, None)
        init_state = decoder.init_state(enc_outputs, None)

        # Each beam: (seq_list, score)
        beams = [([bos_idx], 0.0, init_state)]

        for _ in range(max_len):
            all_candidates = []
            for seq, score, state in beams:
                last_tok = seq[-1]
                if last_tok == eos_idx:
                    # already finished, keep as-is
                    all_candidates.append((seq, score, state))
                    continue

                seq_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
                with torch.no_grad():
                    outputs, _ = decoder(seq_tensor, state)
                    logits = outputs[:, -1, :].squeeze(0)
                    log_probs = torch.log_softmax(logits, dim=-1)

                topk = torch.topk(log_probs, k=beam_size)
                topk_ids = topk.indices.tolist()
                topk_vals = topk.values.tolist()

                for tok_id, logp in zip(topk_ids, topk_vals):
                    new_seq = seq + [int(tok_id)]
                    new_score = score + float(logp)
                    all_candidates.append((new_seq, new_score, state))

            # keep top beam_size
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            beams = ordered

            # stop if all beams end with eos
            if all(b[0][-1] == eos_idx for b in beams):
                break

        # choose best sequence (highest score)
        best_seq = beams[0][0]
        # remove initial bos
        best_seq = best_seq[1:]
        # pad or trim to max_len
        if len(best_seq) < max_len:
            best_seq = best_seq + [0] * (max_len - len(best_seq))
        else:
            best_seq = best_seq[:max_len]

        results.append(torch.tensor(best_seq, dtype=torch.long, device=device).unsqueeze(0))

    return torch.cat(results, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_examples', type=int, default=10, help='number of examples to evaluate and compute BLEU')
    args = parser.parse_args()

    ckpt, epoch = find_latest_checkpoint('checkpoints')
    if not ckpt:
        print('没有找到检查点（checkpoints 目录）。请先训练并保存检查点。')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net.to(device)
    print(f'加载模型: {ckpt}')
    ck = torch.load(ckpt, map_location=device)
    model.load_state_dict(ck.get('model_state_dict', ck))
    model.eval()

    idx2word_t = idx2word_map(vocab_target)
    idx2word_s = idx2word_map(vocab_source)

    bos = vocab_target.get('<bos>')
    eos = vocab_target.get('<eos>')
    if bos is None or eos is None:
        print('词表中缺少 <bos> 或 <eos> 标记，无法生成。')
        return

    # 收集训练集前 N 个样本
    num_examples = args.num_examples
    examples = []
    for i, batch in enumerate(dataloader):
        src, tgt = batch
        for j in range(src.shape[0]):
            examples.append((src[j].unsqueeze(0), tgt[j].unsqueeze(0)))
            if len(examples) >= num_examples:
                break
        if len(examples) >= num_examples:
            break

    if not examples:
        print('数据集中没有样本。')
        return

    hyps = []
    refs = []
    for i, (src, tgt) in enumerate(examples, 1):
        src = src.to(device)
        tgt = tgt.to(device)
        # use beam search for prediction
        pred_idx = beam_search_decode(model, src, max_len=src.shape[1], bos_idx=bos, eos_idx=eos, device=device, beam_size=5)

        # 转为可读文本
        src_words = [idx2word_s.get(int(x), '<unk>') for x in src[0].cpu().tolist()]
        tgt_words = [idx2word_t.get(int(x), '<unk>') for x in tgt[0].cpu().tolist()]
        pred_words = []
        for x in pred_idx[0].cpu().tolist():
            w = idx2word_t.get(int(x), '<unk>')
            if w == '<eos>':
                break
            if w not in ('<pad>', '<bos>'):
                pred_words.append(w)
        # append for BLEU
        hyps.append(pred_words)
        ref_words = [w for w in tgt_words if w not in ('<pad>', '<bos>', '<eos>')]
        refs.append([ref_words])

        print(f"\n=== Sample {i} ===")
        print('Source :', ' '.join([w for w in src_words if w not in ('<pad>', '<bos>', '<eos>')]))
        print('Target :', ' '.join([w for w in tgt_words if w not in ('<pad>', '<bos>', '<eos>')]))
        print('Pred   :', ' '.join(pred_words))

    # compute BLEU over collected examples
    try:
        bleu_score = corpus_bleu(refs, hyps)
        print(f"\nCorpus BLEU for {len(hyps)} examples: {bleu_score:.4f}")
    except Exception as e:
        print(f"BLEU computation failed: {e}")


if __name__ == '__main__':
    main()
