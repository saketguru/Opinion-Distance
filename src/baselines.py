from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from nltk.corpus import stopwords
import io

from gensim.corpora.dictionary import Dictionary
from numpy import zeros, sqrt, sum as np_sum, double
import codecs
import time

total = 0


def bert_baselines(files, result_file):
    # Using BERT from https://github.com/hanxiao/bert-as-service (Pretrained model: uncased_L-12_H-768_A-12)
    from bert_serving.client import BertClient
    bc = BertClient()

    X = np.zeros((len(files), 768))
    start_time = time.time()

    for index, file in enumerate(files):
        doc = codecs.open(file, encoding="utf-8").read().strip()
        X[index:] = bc.encode([doc])
    result = cosine_distances(X)
    end_time = time.time()

    print("Bert Time difference", (end_time - start_time))
    np.savetxt(result_file, result, fmt='%.3f')
    return result


def tf_idf_baseline(files, result_file, input_type):
    print("Baseline of tf-idf")

    start_time = time.time()
    stop_words = [word for word in stopwords.words('english')]
    tfidf = TfidfVectorizer(input=input_type, stop_words=stop_words)
    docs_in_tf_idf = tfidf.fit_transform(files)
    result = cosine_distances(docs_in_tf_idf)
    end_time = time.time()

    print("TF-IDF Time difference ", (end_time - start_time))
    np.savetxt(result_file, result, fmt='%.3f')

    return result


def preprocess_for_wmd(file, model, file_words_dict):
    if file in file_words_dict:
        return file_words_dict[file]
    file_content = io.open(file, 'r', encoding='utf-8').readlines()

    stop_words = [word for word in stopwords.words('english')]
    words_in_file = []

    for line in file_content:
        for word in line.split():
            if word not in stop_words and word in model:
                words_in_file.append(word)

    file_words_dict[file] = words_in_file
    return words_in_file


def wmdistance(document1, document2, model):
    '''
    this function is copied from keyedvectors.wmdistance.
    The reason for copy is: word2vec embeddings contain non lower cases entry, so we convert all entries to lowercase
    and store in a dictionary. The only change below function does is to change that dictionary.
    :param document1:
    :param document2:
    :param model:
    :return:
    '''
    # Remove out-of-vocabulary words.
    len_pre_oov1 = len(document1)
    len_pre_oov2 = len(document2)
    document1 = [token for token in document1 if token in model]
    document2 = [token for token in document2 if token in model]
    diff1 = len_pre_oov1 - len(document1)
    diff2 = len_pre_oov2 - len(document2)

    if len(document1) == 0 or len(document2) == 0:
        print("At least one of the documents had no words that werein the vocabulary. "
              "Aborting (returning inf).")
        return float('inf')

    dictionary = Dictionary(documents=[document1, document2])
    vocab_len = len(dictionary)

    if vocab_len == 1:
        # Both documents are composed by a single unique token
        return 0.0

    # Sets for faster look-up.
    docset1 = set(document1)
    docset2 = set(document2)

    # Compute distance matrix.
    distance_matrix = zeros((vocab_len, vocab_len), dtype=double)
    for i, t1 in dictionary.items():
        for j, t2 in dictionary.items():
            if t1 not in docset1 or t2 not in docset2:
                continue
            # Compute Euclidean distance between word vectors.
            distance_matrix[i, j] = sqrt(np_sum((np.asarray(model[t1]) - np.asarray(model[t2])) ** 2))

    if np_sum(distance_matrix) == 0.0:
        # `emd` gets stuck if the distance matrix contains only zeros.
        print('The distance matrix is all zeros. Aborting (returning inf).')
        return float('inf')

    def nbow(document):
        d = zeros(vocab_len, dtype=double)
        nbow = dictionary.doc2bow(document)  # Word frequencies.
        doc_len = len(document)
        for idx, freq in nbow:
            d[idx] = freq / float(doc_len)  # Normalized word frequencies.
        return d

    # Compute nBOW representation of documents.
    d1 = nbow(document1)
    d2 = nbow(document2)

    from pyemd import emd

    # Compute WMD.
    return emd(d1, d2, distance_matrix)


def wmd_baseline(files, result_file, model):
    print("Word mover baseline")

    start_time = time.time()

    file_words_dict = {}
    X = np.zeros((len(files), len(files)))
    polarity_distance_fp = codecs.open(result_file, "w")

    for i, file1 in enumerate(files):
        words_in_file1 = preprocess_for_wmd(file1, model, file_words_dict)

        for j, file2 in enumerate(files):

            words_in_file2 = preprocess_for_wmd(file2, model, file_words_dict)

            if len(words_in_file1) == 0 or len(words_in_file2) == 0:
                polarity_distance_fp.write("%.5f" % 1.0 + " ")
                X[i][j] = 1
                continue
            distance = wmdistance(words_in_file1, words_in_file2, model)
            polarity_distance_fp.write("%.5f" % distance + " ")
            X[i][j] = distance

        polarity_distance_fp.write("\n")

        polarity_distance_fp.flush()
    polarity_distance_fp.close()

    end_time = time.time()
    X = X / np.nanmax(X)
    np.savetxt(result_file, X, fmt='%.5f', delimiter='\t')
    print("Time difference", (end_time - start_time))

    return X


def doc2vec_baseline(files, result_file, model):
    X = np.zeros((len(files), 200))
    start_time = time.time()

    for index, file in enumerate(files):
        doc = codecs.open(file, encoding="utf-8").read().strip()
        X[index:] = model.infer_vector(doc)
    result = cosine_distances(X)
    end_time = time.time()

    print("Doc2vec Time difference", (end_time - start_time))
    np.savetxt(result_file, result, fmt='%.3f')
    return result
