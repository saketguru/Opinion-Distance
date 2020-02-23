import codecs
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedShuffleSplit
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils import read_distance_matrix_from_file, get_files_and_ground_truth
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import warnings

warnings.filterwarnings('ignore')

seed_n = 42
np.random.seed(seed_n)

bog_features = 100
tf_idf_features = 50
lsa_features = 20


def preProcess(s):
    return ps.stem(s)


def model(clf, features_train, features_test, labels_train, labels_test):
    clf.fit(features_train, np.ravel(labels_train))
    pred = clf.predict(features_test)
    return f1_score(np.ravel(labels_test), pred, average='weighted')


def bag_of_words_uni(files, features):
    BOG_generator = CountVectorizer(input="filename", stop_words=stop_wordss, max_features=bog_features,
                                    analyzer='word', ngram_range=(1, 1), lowercase=True, preprocessor=preProcess)

    generator = BOG_generator
    generator.fit(files)
    bog = generator.transform(files).todense()
    return np.hstack((features, bog))


def bag_of_words_uni_bi(files, features):
    BOG_generator = CountVectorizer(input="filename", stop_words=stop_wordss, max_features=bog_features,
                                    analyzer='word', ngram_range=(1, 2), lowercase=True, preprocessor=preProcess)

    generator = BOG_generator
    generator.fit(files)
    bog = generator.transform(files).todense()
    return np.hstack((features, bog))


def add_sentiment_polarity(files, features):
    analyser = SentimentIntensityAnalyzer()
    polarity_scores = np.zeros((features.shape[0], 1))
    for index, file in enumerate(files):
        polarity_scores[index] = analyser.polarity_scores(codecs.open(file, encoding="utf-8").read())['compound']

    return np.hstack((features, polarity_scores))


def text_similarity_measure(files, features):
    TF_IDF_generator = TfidfVectorizer(input="filename", stop_words=stop_wordss, max_features=tf_idf_features,
                                       analyzer='word', ngram_range=(1, 2), lowercase=True, preprocessor=preProcess,
                                       max_df=0.8, min_df=2)

    tf_generator = TF_IDF_generator.fit(files)
    tf_idf = tf_generator.transform(files).todense()
    svd = TruncatedSVD(lsa_features)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    X_train_lsa = lsa.fit_transform(tf_idf)
    return np.hstack((features, X_train_lsa))


def use_opinion_distance_features(features):
    filename = path + opinion_name
    opinion_distance = read_distance_matrix_from_file(filename)
    features = np.hstack((features, opinion_distance))
    return features


def distance_based_features(features, file):
    tf_idf_distance = read_distance_matrix_from_file(path + file)
    features = np.hstack((features, tf_idf_distance))
    return features


def generate_features(files):
    features = np.zeros((len(files), 1))

    if unigram_individual:
        features = bag_of_words_uni(files, features)

    if bigram_individual:
        features = bag_of_words_uni_bi(files, features)

    if lsa_individual:
        features = text_similarity_measure(files, features)

    if senti_individual:
        features = add_sentiment_polarity(files, features)

    if tf_idf_individual:
        features = distance_based_features(features, tf_idf_name)

    if wmd_individual:
        features = distance_based_features(features, wmd_name)

    if doc2vec_individual:
        features = distance_based_features(features, doc2vec_name)

    if bert_individual:
        features = distance_based_features(features, bert_name)

    if sent2vec_individual:
        features = distance_based_features(features, sent2vec_name)

    if opinion_individual:
        features = use_opinion_distance_features(features)

    features = np.delete(features, 0, -1)

    return features


def get_accuracy(full_path_files, labels):
    X = generate_features(full_path_files)
    y = np.asarray(labels) + 1

    # print("Generated features")

    skf = StratifiedShuffleSplit(n_splits=3, test_size=0.3)
    results = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ros = RandomOverSampler()
        X_train, y_train = ros.fit_sample(X_train, y_train)

        param_grid = [{'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.001], 'kernel': ['rbf', 'linear']}]
        clf = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1_weighted')

        results.append(model(clf, X_train, X_test, y_train, y_test))

    avg = 0.0
    for tup in results:
        avg += tup

    return avg / len(results)


def run():
    print("Supervised Opinion Distance")
    feature_names = {
        "Unigram": ['unigram_individual'],
        "Bigram": ['bigram_individual'],
        "LSA": ['lsa_individual'],
        "Sentiment": ['senti_individual'],
        "Bigram + Sentiment": ['bigram_individual', 'senti_individual'],
        "TF_IDF": ['tf_idf_individual'],
        # "WMD": ['wmd_individual'],
        # "Sent2vec": ['sent2vec_individual'],
        # "Doc2vec": ['doc2vec_individual'],
        # "Bert": ['bert_individual'],
        "Unigram + Bigrams": ['unigram_individual', 'bigram_individual'],
        "Unigram + Sentiment": ['unigram_individual', 'senti_individual'],
        "Unigram + LSA": ['unigram_individual', 'lsa_individual'],
        "Unigram + TF_IDF": ['unigram_individual', 'tf_idf_individual'],
        # "Unigram + WMD": ['unigram_individual', 'wmd_individual'],
        # "Unigram + Sent2vec": ['unigram_individual', 'sent2vec_individual'],
        # "Unigram + Doc2vec": ['unigram_individual', 'doc2vec_individual'],
        # "Unigram + BERT": ['unigram_individual', 'bert_individual'],
        "Opinion Distance": ['opinion_individual'],
        "Opinion Distance + Unigram": ['opinion_individual', 'unigram_individual'],
        "Opinion Distance + Bigrams": ['opinion_individual', 'bigram_individual'],
        "Opinion Distance + Sentiment": ['opinion_individual', 'senti_individual'],
        "Opinion Distance + LSA": ['opinion_individual', 'lsa_individual'],
        # "Opinion Distance + WMD": ['opinion_individual', 'wmd_individual'],

    }

    results = dict()
    for key, flags in feature_names.items():
        for flag in flags:
            globals()[flag] = True

        print("Running for %s" % key)

        f1_weighted = get_accuracy(full_path_files, labels)
        results[key] = f1_weighted

        for flag in flags:
            globals()[flag] = False

    print("\n\n")
    for key, val in results.items():
        print(key, val)


if __name__ == "__main__":
    path = sys.argv[1]
    # print(path)

    tf_idf_name = "dist_mats/tf_idf.txt"
    wmd_name = "dist_mats/wmd.txt"
    doc2vec_name = "dist_mats/doc2vec.txt"
    sent2vec_name = "dist_mats/sent2vec.txt"
    bert_name = "dist_mats/bert.txt"
    opinion_name = ("dist_mats/OD_embedding_strategy_%s.txt") % ("word2vec")

    ps = PorterStemmer()
    stop_wordss = [word for word in stopwords.words('english')]

    unigram_individual, bigram_individual, lsa_individual, \
    senti_individual, tf_idf_individual, wmd_individual, \
    doc2vec_individual, sent2vec_individual, bert_individual, opinion_individual = (False, False, False, False, False,
                                                                                    False, False, False, False, False)

    if "CiviQ_Seanad" in path:
        Aspect = ["Abolish", "Do not Abolish", "Democracy"]
        full_path_files, labels = get_files_and_ground_truth(path, Aspect, docs_path="docs")

    run()
