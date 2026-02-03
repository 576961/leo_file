

with open('C:/Users/Lenovo/Desktop/deu.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
print(1)
def preprocess_nmt(text):
    """预处理数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) >= 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

def pad_nmt(text, max_len):
    for i in range(len(text)):
        if len(text[i]) + 2 >= max_len:
            text[i] = ['<bos>'] + text[i][:max_len-2] + ['<eos>']
        else:
            text[i] = ['<bos>'] + text[i] + ['<pad>']*(max_len-len(text[i])-2) + ['<eos>']
    return text


def create_vocab(text):
    num = 0
    dic_count = {}
    for sen in text:
        for word in sen:
            if word not in dic_count:
                dic_count[word] = 1
            else:
                dic_count[word] += 1

    for sen in text:
        for i in range(len(sen)):
            if dic_count[sen[i]] < 2:
                sen[i] = '<unk>'
    vocab = {}
    for sen in text:
        for word in sen:
            if word not in vocab:
                vocab[word] = num
                num += 1
    return vocab

def build(text, vocab):
    for sen in text:
        for i in range(len(sen)):
            sen[i] = vocab[sen[i]]
    return text

text = preprocess_nmt(raw_text)
source, target = tokenize_nmt(text)
source = pad_nmt(source, max_len=10)
target = pad_nmt(target, max_len=10)
vocab_source = create_vocab(source)
vocab_target = create_vocab(target)
source_num = build(source, vocab_source)
target_num = build(target, vocab_target)

print(source_num[:50])
print(target_num[:50])
print(len(vocab_source))
print(len(vocab_target))


