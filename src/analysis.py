import pprint
import logging
import collections
from gensim import models
from gensim import corpora
from collections import defaultdict
from gensim import similarities
from gensim.test.utils import datapath
import pyLDAvis.gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
paperName = ['renmin', 'xinhua', 'huanqiu', 'guancha', 'wenhui', 'chinadaily', 'sputniknews', 'BBC', 'DW', 'WSJ', 'NTY']
paperSize = [158,       244,      283,       188,       170,      203,          317,           244,   493,  687,   431]
paperPage = [30,        26,       31,        28,        26,       23,           32,            29,    1,    35,    56]


def read_data():
    corpus = []
    articles = words = 0

    fin = open('data\\xinhua.txt','r')
    lines = fin.readlines()
    articles += len(lines)
    words += sum([len(line.split(' ')) for line in lines])
    print('xinhua', len(lines), sum([len(line.split(' ')) for line in lines]))
    corpus += [line[:-1] for line in lines]

    fin = open('data\\renmin.txt','r')
    lines = fin.readlines()
    articles += len(lines)
    words += sum([len(line.split(' ')) for line in lines])
    print('renmin', len(lines), sum([len(line.split(' ')) for line in lines]))
    corpus += [line[:-1] for line in lines]

    fin = open('data\\huanqiu.txt','r')
    lines = fin.readlines()
    articles += len(lines)
    words += sum([len(line.split(' ')) for line in lines])
    print('huanqiu', len(lines), sum([len(line.split(' ')) for line in lines]))
    corpus += [line[:-1] for line in lines]

    fin = open('data\\guancha.txt','r')
    lines = fin.readlines()
    articles += len(lines)
    words += sum([len(line.split(' ')) for line in lines])
    print('guancha', len(lines), sum([len(line.split(' ')) for line in lines]))
    corpus += [line[:-1] for line in lines]

    fin = open('data\\wenhui.txt','r')
    lines = fin.readlines()
    articles += len(lines)
    words += sum([len(line.split(' ')) for line in lines])
    print('wenhui', len(lines), sum([len(line.split(' ')) for line in lines]))
    corpus += [line[:-1] for line in lines]

    fin = open('data\\chinadaily.txt','r')
    lines = fin.readlines()
    articles += len(lines)
    words += sum([len(line.split(' ')) for line in lines])
    print('chinadaily', len(lines), sum([len(line.split(' ')) for line in lines]))
    corpus += [line[:-1] for line in lines]

    fin = open('data\\sputniknews.txt','r')
    lines = fin.readlines()
    articles += len(lines)
    words += sum([len(line.split(' ')) for line in lines])
    print('sputniknews', len(lines), sum([len(line.split(' ')) for line in lines]))
    corpus += [line[:-1] for line in lines]

    chineseArticles = articles
    chineseWords = words
    print(chineseArticles, chineseWords)

    fin = open('data\\BBC.txt','r')
    lines = fin.readlines()
    articles += len(lines)
    words += sum([len(line.split(' ')) for line in lines])
    print('BBC', len(lines), sum([len(line.split(' ')) for line in lines]))
    corpus += [line[:-1] for line in lines]

    fin = open('data\\DW.txt','r')
    lines = fin.readlines()
    articles += len(lines)
    words += sum([len(line.split(' ')) for line in lines])
    print('DW', len(lines), sum([len(line.split(' ')) for line in lines]))
    corpus += [line[:-1] for line in lines]

    fin = open('data\\WSJ.txt','r')
    lines = fin.readlines()
    articles += len(lines)
    words += sum([len(line.split(' ')) for line in lines])
    print('WSJ', len(lines), sum([len(line.split(' ')) for line in lines]))
    corpus += [line[:-1] for line in lines]

    fin = open('data\\NTY.txt','r')
    lines = fin.readlines()
    articles += len(lines)
    words += sum([len(line.split(' ')) for line in lines])
    print('NYT', len(lines), sum([len(line.split(' ')) for line in lines]))
    corpus += [line[:-1] for line in lines]

    westernArticles = articles-chineseArticles
    westernWords = words-chineseWords
    print(westernArticles, westernWords)

    print(articles, words)
    return [document.split(' ') for document in corpus]


def prepare_data(corpus):
    # frequency = defaultdict(int)
    # for document in corpus:
    #     for token in document:
    #         frequency[token] += 1
    # processedCorpus = [[token for token in document if frequency[token] > 1] for document in corpus]
    # dictionary = corpora.Dictionary(processedCorpus)
    # bowCorpus = [dictionary.doc2bow(document) for document in processedCorpus]
    dictionary = corpora.Dictionary(corpus)
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    bowCorpus = [dictionary.doc2bow(document) for document in corpus]

    print('Number of unique tokens %d' % len(dictionary))
    print('Number of documents %d' % len(bowCorpus))

    return dictionary, bowCorpus

    tfidf = models.TfidfModel(bowCorpus)
    # print(tfidf[bowCorpus[0]])
    index = similarities.SparseMatrixSimilarity(tfidf[bowCorpus[1374:]], num_features=45290)
    queryBow = dictionary.doc2bow(['中国', '疫情'])
    sims = index[tfidf[queryBow]]

    filterNum = [document+1374 for document, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:1300]]
    bowResults = bowCorpus[:1374] + [bowCorpus[document] for document in filterNum]

    # corpusResult = processedCorpus[:1374] + [processedCorpus[document] for document in filterNum]
    # print(len(corpusResult))
    # print(sum([len(document) for document in corpusResult]))

    # verifi = processedCorpus[:1374]
    # bowResults = bowCorpus[:1374]
    # num = 0
    # for document, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:1300]:
        # num += 1
        # print(document, score)
    #     bowResults.append(bowCorpus[document+1374])
    #     verifi.append(processedCorpus[document+1374])
    # print(num)
    # print(len(bowResults))
    return dictionary, bowResults


def model_lda(dictionary, bowCorpus):
    numTopics = 7
    chunksize = 3418
    passes = 20
    iterations = 400
    evalEvery = None
    alpha = 'auto'
    eta = 'auto'
    temp = dictionary[0]
    id2word = dictionary.id2token

    # print(len(id2word))
    model = models.LdaModel(
        corpus=bowCorpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha=alpha,
        eta=eta,
        iterations=iterations,
        num_topics=numTopics,
        passes=passes,
        eval_every=evalEvery
    )
    tempFile = datapath("F:\\learning\\Fangfang\\lda"+str(numTopics)+"-2")
    model.save(tempFile)
    # topTopics = model.top_topics(bowCorpus)
    # pprint.pprint(topTopics)
    vis = pyLDAvis.gensim.prepare(model, bowCorpus, dictionary)
    pyLDAvis.save_html(vis, 'lda_'+ str(numTopics) +'-2.html')


def compare_data(dictionary, bowCorpus, fangfang, modelFile):
    papers = dict()
    st, ed = 0, 0
    for index, paper in enumerate(paperName):
        ed += paperSize[index]
        papers[paper] = bowCorpus[st:ed]
        st = ed
    # print(len(papers['renmin'] + papers['xinhua'] + papers['huanqiu'] + papers['guancha'] + papers['wenhui'] + papers['chinadaily'] + papers['sputniknews'] + papers['BBC'] + papers['DW'] + papers['WSJ'] + papers['NYT']))
    tempFile = datapath(modelFile)
    model = models.LdaModel.load(tempFile)
    # result = model.top_topics(bowCorpus)
    # pprint.pprint(result)
    for index, paper in enumerate(paperName):
        topicsSum = [0.0 for i in range(7)]
        for result in model.get_document_topics(papers[paper]):
            for (topic, value) in result:
                topicsSum[topic] += value
        avgTopics = [value/paperSize[index] for value in topicsSum]
        print(paper, avgTopics)
        # pprint.pprint(result)
    bowFangfang = [dictionary.doc2bow(document.split(' ')) for document in fangfang]
    print(len(bowFangfang))
    topicsSum = [0.0 for i in range(7)]
    for result in model.get_document_topics(bowFangfang):
        for (topic, value) in result:
            topicsSum[topic] += value
    avgTopics = [value/60 for value in topicsSum]
    print('fangfang', avgTopics)


def model_doc2vec(corpus, fangfang):
    # print(corpus)
    # print(fangfang)
    trainCorpus = []
    for i, document in enumerate(corpus):
        trainCorpus.append(models.doc2vec.TaggedDocument(document, [i]))
    # for i, document in enumerate(fangfang):
    #     trainCorpus.append(models.doc2vec.TaggedDocument(document, [i + 3112]))
    # temp = []
    # for document in fangfang:
    #     temp += document
    # print(temp)
    # trainCorpus.append(models.doc2vec.TaggedDocument(temp, [3112 + 60]))
    # print(trainCorpus[-1])
    modelDoc2vec = models.doc2vec.Doc2Vec(vector_size=40, min_count=2, epochs=40)
    modelDoc2vec.build_vocab(trainCorpus)
    # print(modelDoc2vec.wv.vocab['武汉'].count)
    modelDoc2vec.train(trainCorpus, total_examples=modelDoc2vec.corpus_count, epochs=modelDoc2vec.epochs)
    ranks = []
    for docId in range(len(trainCorpus)):
        vector = modelDoc2vec.infer_vector(trainCorpus[docId].words)
        sims = modelDoc2vec.docvecs.most_similar([vector], topn=len(modelDoc2vec.docvecs))
        rank = [docid for docid, sim in sims].index(docId)
        ranks.append(rank)

    counter = collections.Counter(ranks)
    print(counter)
    tempFile = datapath("F:\\learning\\Fangfang\\doc2vec3")
    modelDoc2vec.save(tempFile)


def compare_distance(corpus, fangfang, fileName):
    tempFile = datapath(fileName)
    modelDoc2vec = models.doc2vec.Doc2Vec.load(tempFile)
    temp = []
    for document in fangfang:
        temp += document
    vector = modelDoc2vec.infer_vector(temp)
    sims = modelDoc2vec.docvecs.most_similar([vector], topn=len(modelDoc2vec.docvecs))
    # print(sims[:10])
    result = [docId for docId, sim in sims]
    print(result)
    print([corpus[docId][:12] for docId in result[:10]])
    return result


corpus = read_data()

fin = open('data\\fangfang.txt','r', encoding='utf-8')
lines = fin.readlines()
fangfang = [line[:-1].split(' ') for line in lines]
# print(fangfang)

# dictionary, bowCorpus = prepare_data(corpus)
# model_lda(dictionary, bowCorpus)
# compare_data(dictionary, bowCorpus, fangfang, 'F:\\learning\\Fangfang\\lda7')
# model_doc2vec(corpus, fangfang)
# headlines = compare_distance(corpus, fangfang, "F:\\learning\\Fangfang\\doc2vec3")
