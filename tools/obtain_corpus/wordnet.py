import os        
from nltk.corpus import wordnet as wn

def organize_nouns_by_category():
    # 创建一个字典来存储按类别分组的名词
    categories = {}

    # 获取所有名词的同义词集
    for synset in wn.all_synsets(wn.NOUN):
        # 获取同义词集的词汇类别
        lexname = synset.lexname()
        
        # 如果该类别尚未在字典中，则添加它
        if lexname not in categories:
            categories[lexname] = set()
        
        # 将同义词集中的所有词（lemma）添加到相应的类别中
        for lemma in synset.lemmas():
            categories[lexname].add(lemma.name())

    return categories

def write_categories_to_files(categories):
    # 确保输出目录存在
    output_dir = 'wordnet_nouns_by_category'
    os.makedirs(output_dir, exist_ok=True)

    # 为每个类别写入一个文件
    for category, nouns in categories.items():
        with open(os.path.join(output_dir, f"{category}.txt"), 'w') as file:
            for noun in sorted(nouns):
                file.write(noun + '\n')

# 主程序
categories = organize_nouns_by_category()
write_categories_to_files(categories)




