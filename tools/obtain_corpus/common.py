import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk import pos_tag

# 打开输入文件和两个输出文件
with open('20k_common_words.txt', 'r') as words_file, \
     open('noun.txt', 'w') as noun_file, \
     open('adj.txt', 'w') as adj_file:

    # 逐行读取单词
    for word in words_file:
        word = word.strip()  # 去掉可能的空格和换行符
        if word:  # 如果行不为空
            # 词性标注
            tagged_word = pos_tag([word])
            pos = tagged_word[0][1]  # 获取词性

            # 根据词性分类存储
            if pos.startswith('NN'):  # 名词
                noun_file.write(word + '\n')
            elif pos.startswith('JJ'):  # 形容词
                adj_file.write(word + '\n')
            # 其他词性则忽略
