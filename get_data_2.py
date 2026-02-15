
from datasets import load_dataset

ds = load_dataset("wmt14", "de-en", split="train")

with open("train.en", "w", encoding="utf-8") as f_en, \
     open("train.de", "w", encoding="utf-8") as f_de:
    
    for item in ds:
        en = item["translation"]["en"]
        de = item["translation"]["de"]
        
        # 过滤：两门语言均非空且不是仅空白字符
        if en and de and en.strip() and de.strip():
            # 确保句子内部换行符不会破坏行对应关系
            f_en.write(en.replace("\n", " ") + "\n")
            f_de.write(de.replace("\n", " ") + "\n")
        else:
            # 可选：打印被跳过的样本索引（便于核对）
            # print(f"Skipped index {i}: en='{en}', de='{de}'")
            pass
