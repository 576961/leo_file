import argparse
import os
import torch

from transformer_new import net, tokenizer, seq_len


def greedy_decode(model, src_batch, max_len, bos_idx, eos_idx, device):
    encoder = model.encoder
    decoder = model.decoder

    enc_outputs = encoder(src_batch, None)
    state = decoder.init_state(enc_outputs, None)

    batch_size = src_batch.shape[0]
    generated = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)

    for _ in range(max_len):
        outputs, state = decoder(generated, state)
        next_logits = outputs[:, -1, :]
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if ((next_token == eos_idx).all()):
            break

    return generated[:, 1:]


def load_checkpoint(path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ck = torch.load(path, map_location=device)
    # common keys used in this repo: 'model_state' or 'model_state_dict'
    state = ck.get('model_state', ck.get('model_state_dict', ck))
    model = net.to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints_new/transformer_step_10.pth', help='checkpoint path')
    parser.add_argument('--text', required=True, help='English sentence to translate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_checkpoint(args.ckpt, device)

    # tokenizer and special ids
    tok = tokenizer
    bos_id = getattr(tok, 'cls_token_id', None) or getattr(tok, 'bos_token_id', None)
    eos_id = getattr(tok, 'sep_token_id', None) or getattr(tok, 'eos_token_id', None)
    if bos_id is None or eos_id is None:
        # fallback to common token strings
        bos_id = tok.convert_tokens_to_ids('[CLS]') if '[CLS]' in tok.get_vocab() else 0
        eos_id = tok.convert_tokens_to_ids('[SEP]') if '[SEP]' in tok.get_vocab() else 2

    # encode input sentence
    enc = tok(args.text, padding='max_length', truncation=True, max_length=seq_len, return_tensors='pt')
    src = enc['input_ids'].to(device)

    with torch.no_grad():
        pred_idx = greedy_decode(model, src, max_len=seq_len, bos_idx=bos_id, eos_idx=eos_id, device=device)

    # Convert to list and trim at eos
    pred_list = pred_idx[0].cpu().tolist()
    out_ids = []
    for id_ in pred_list:
        if id_ == eos_id:
            break
        out_ids.append(id_)

    # decode using tokenizer (skip special tokens)
    try:
        text_out = tok.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    except Exception:
        # fallback: map ids to tokens then join
        toks = [tok._convert_id_to_token(i) for i in out_ids]
        text_out = ' '.join(toks)

    print(text_out)


if __name__ == '__main__':
    main()
