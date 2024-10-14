import csv

# 定义名词和形容词的词性标记
noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
adjective_tags = {'JJ', 'JJR', 'JJS'}

# 打开CSV文件
with open('words_pos.csv', 'r', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file)

    # 打开两个txt文件，一个用于写入名词，另一个用于写入形容词
    with open('nouns.txt', 'w', encoding='utf-8') as nouns_file, \
            open('adjectives.txt', 'w', encoding='utf-8') as adjectives_file:

        # 遍历CSV文件的每一行
        for row in reader:
            # 检查第三列的词性标记
            if row[2] in noun_tags:
                # 写入名词文件
                nouns_file.write(row[1] + '\n')
            elif row[2] in adjective_tags:
                # 写入形容词文件
                adjectives_file.write(row[1] + '\n')
