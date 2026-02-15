
from transformers import AutoTokenizer
'''
# 从 HuggingFace Hub 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en")

# 保存到本地目录（例如 "./my_bert2bert_tokenizer"）
save_directory = "./my_bert2bert_tokenizer"
tokenizer.save_pretrained(save_directory)
print(1)'''

tokenizer = AutoTokenizer.from_pretrained("./my_bert2bert_tokenizer")
print(2)
print(tokenizer.unk_token)       # 应输出 '[UNK]'
print(tokenizer.unk_token_id)    # 应输出一个整数（例如 100）

with open(r'C:\Users\Lenovo\PycharmProjects\ProjectTransformer\train.en.preprocessed.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.split('\n')[:5]  # 打印前5行文本以验证读取正确

#L = []

X = tokenizer(text, padding=True, truncation=True, max_length=20, return_tensors="pt")['input_ids']
print(X[0])
print(X.shape)

#print(L)

# 获取词表字典 {token: id}
vocab_dict = tokenizer.get_vocab()

print(f"词表大小: {len(vocab_dict)}")
# 输出示例: 词表大小: 32000（或其他数值）

# 查看前几个词条
print(list(vocab_dict.items())[:10])