import logging
import re
import jieba
from pprint import pprint
from gensim.summarization import summarize
from gensim.summarization import keywords

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
jieba.add_word("新冠肺炎")
jieba.add_word("冠性肺炎")
jieba.add_word("新型冠性肺炎")
jieba.add_word("新型冠状病毒")
jieba.add_word("火神山")
jieba.add_word("雷神山")
jieba.add_word("方舱医院")
jieba.add_word("人民日报")
jieba.add_word("人民网")
jieba.add_word("卫星通讯社")
jieba.add_word("谭德塞")
jieba.add_word("新华网")
jieba.add_word("新华社")
jieba.add_word("武汉肺炎")
jieba.add_word("中国病毒")
jieba.add_word("武汉病毒")
jieba.add_word("中国肺炎")
jieba.add_word("钻石公主号")
jieba.add_word("华南海鲜市场")


def fangfang_abstract():
    fin = open('F:\\learning\\Fangfang\\originalData\\fangfang.txt', 'r', encoding='utf-8')
    corpus = []
    doc = []
    # temp = []
    # temp2 = []
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    for line in fin.readlines():
        if '*' in line:
            doc.append('')
            corpus.append('。. '.join(doc))
            doc = []
            continue
        sentences = re.split('。|；|？|!', line[:-1])
        tmp = re.sub(pattern, '', line)
        if len(tmp) == 0:
            continue
        # print(len(tmp), line[:-1])
        # temp.append(len(tmp))
        # temp2.append(tmp)
        # print(sentences)
        for sentence in sentences:
            result = ' '.join(jieba.cut(sentence.strip()))
            if len(result) > 0:
                doc.append(result)
        # print('//================')
    # print(len(corpus))
    # print(corpus[0])
    # print(temp2)
    # print(temp)
    corpus.reverse()
    fout = open('F:\\learning\\Fangfang\\data\\fangfangAbstract.txt', 'w', encoding='utf-8')
    totalDoc = ''
    for i in range(60):
        totalDoc = totalDoc + corpus[i]
    # print(totalDoc)
    text = summarize(totalDoc, ratio=0.02)
    text = text.replace(' ', '')
    text = text.replace('.', '')
    fout.write("全文摘要：\n")
    fout.write(''.join(text.split('\n')))
    fout.write('\n\n')
    text = keywords(totalDoc, ratio=0.2)
    fout.write("关键词:\n")
    fout.write(' '.join(text.split('\n')))
    fout.write("\n\n")
    for i in range(60):
        ratio = 500 / len(corpus[i])
        text = summarize(corpus[i], ratio=ratio)
        text = text.replace(' ', '')
        text = text.replace('.', '')
        print(i, ratio)
        # print(''.join(text.split('\n')))
        fout.write(str(i)+'\n')
        fout.write(''.join(text.split('\n')))
        fout.write('\n\n')


fangfang_abstract()