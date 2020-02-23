from gensim.models import KeyedVectors
import numpy as np
import codecs
from sklearn.metrics.pairwise import cosine_distances
import ast
import os
import re

from gensim.models.deprecated.doc2vec import Doc2Vec as doc2vec_dep

stored_embeddings_dict = dict()


class Model:

    def __init__(self, embedding_strategy, path=None, nfp_path=None):
        self.nfp_path = "nfpv2/"
        self.model = dict()
        self.my_additions = dict()

        if embedding_strategy == "word2vec":
            self.model = KeyedVectors.load_word2vec_format("embedding/GoogleNews-vectors-negative300.bin.gz",
                                                           binary=True)

            self.strategy = "word2vec"

        if embedding_strategy == "doc2vec":
            ndocp = doc2vec_dep.load("embedding/doc2vec_filt_existing_3ormore_inlinks_5000_output.bin")
            ndocp.train([], total_examples=ndocp.corpus_count, epochs=ndocp.iter)
            self.model = ndocp

            self.strategy = "doc2vec"

    def read_embeddings_in_model(self, path, method, model):
        files = os.listdir(path + self.nfp_path + method + "_embeddings/")
        for file in files:
            if "embeddings" not in file:
                continue

            f = codecs.open(path + self.nfp_path + method + "_embeddings/" + file, "r", encoding="utf-8")
            for line in f.readlines():
                records = line.split("\t")
                embed = "[ " + records[-1].strip() + " ]"
                if method == "doc2vec":
                    key = clear_signature(records[0].lower())
                else:
                    key = records[0].lower()

                model[key] = ast.literal_eval(embed)

    def __getitem__(self, key, context=None, index=None, strategy=None):
        if strategy is None:
            strategy = self.strategy

        if strategy == "word2vec":
            if key in self.model:
                return self.model[key]
            else:
                if key in self.my_additions:
                    return self.my_additions[key]
                return None

        elif strategy == "doc2vec":
            key = clear_signature(key)
            if key in self.model.wv:
                return self.model[key]
            return self.my_additions[key]

        return None

    def __contains__(self, item):
        if self.strategy == "word2vec":
            if item in self.model:
                return True
            else:
                return False

        if self.strategy == "doc2vec":
            if item in self.model.wv:
                return True
            else:
                return False

    def __setitem__(self, key, words):
        if self.strategy == "word2vec":
            vector = np.zeros((300,))
            for word in words:
                vector = np.add(vector, self.model[word])
            self.my_additions[key] = vector
        elif self.strategy == "doc2vec":
            self.my_additions[key] = self.model.infer_vector(words)

    def add_phrases_to_model(self, phrase, words=None, vector=None):
        if vector is not None:
            self.my_additions[phrase] = vector
            return True
        if self.strategy == "word2vec":
            vector = np.zeros((300,))
            for word in words:
                if word not in self.model:
                    continue
                vector = np.add(vector, self.model[word])
            vector = vector / len(words)
            self.my_additions[phrase] = vector
        elif self.strategy == "doc2vec":
            sentence = " ".join(word for word in words)
            phrase = clear_signature(phrase)
            self.my_additions[phrase] = self.model.infer_vector(sentence)

    def infer_vector(self, doc_words, alpha=0.1, min_alpha=0.0001, steps=5):
        doc_words = clear_signature(doc_words)
        return self.model.infer_vector(doc_words=doc_words)

    def get_difference_between_vecs(self, term1_vec, term2_vec, type):
        if type == "cosine":
            if np.array_equal(term1_vec, term2_vec):
                return 0.00001
            return cosine_distances([term1_vec], [term2_vec])[0][0]

    def compute_semantic_distance(self, term1, term2, type):
        term1_vec = np.asarray(self[term1])
        term2_vec = np.asarray(self[term2])
        dist = self.get_difference_between_vecs(term1_vec, term2_vec, type)
        return dist


def clear_signature(word):
    signature = ".sign.n"
    if '' == word:
        return word
    return re.sub(signature + "[0-9]+", "", word)
