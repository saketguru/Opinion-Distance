from nltk.corpus import stopwords
import nltk
from pyemd import emd, emd_with_flow
from numpy import zeros, double, sqrt, sum as np_sum
from gensim.corpora.dictionary import Dictionary
import numpy as np
import ast
import networkx as nx
import codecs
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import time

semantic_threshold = 0

filepath = None
embedding_strategy = None
lemmatizer = WordNetLemmatizer()


def get_lemmtized_word(word):
    nounpos = ['NN', 'NNS', 'NNP', 'NNPS']
    adjpos = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
    verbpos = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    def get_pos_for_lemmatizer(tag):
        wn_tag = wn.NOUN
        if tag in nounpos:
            wn_tag = wn.NOUN
        if tag in adjpos:
            wn_tag = wn.ADJ
        if tag in verbpos:
            wn_tag = wn.VERB
        return wn_tag

    tag = nltk.tag.pos_tag([word])[0][1]
    wn_tag = get_pos_for_lemmatizer(tag)
    wd = lemmatizer.lemmatize(word, pos=wn_tag)
    return wd


def check_words_in_phrase_and_add_to_model(phrase, model):
    phrase_rep = phrase
    phrase_rep = phrase_rep.replace(" ", "_")

    if phrase_rep in model:
        model.add_phrases_to_model(phrase, vector=model[phrase_rep])
        return True

    words = phrase.split(" ")
    nwords = []
    for word in words:
        if "_" in word:
            nwords.extend(word.split("_"))
        else:
            nwords.append(word)

    words = nwords

    if len(words) == 1:
        return False

    embedding_words = []
    atleast_one_word_in_model = False

    stop_wordss = stopwords.words('english')

    for index, word in enumerate(words):
        if word in stop_wordss:
            continue

        if word in model:
            word = get_lemmtized_word(word)
            embedding_words.append(word)
            atleast_one_word_in_model = True
        elif word.lower() in model:
            word = get_lemmtized_word(word.lower())
            embedding_words.append(word)
            atleast_one_word_in_model = True

    if len(embedding_words) == 0:
        return False

    if atleast_one_word_in_model:
        model.add_phrases_to_model(phrase, words=embedding_words)
        return True
    return False


def preprocess_words(noun_freq_polar, model):
    nfp_lower = []
    for (noun, freq, polarity) in noun_freq_polar:
        freq = int(freq)
        if freq > 0:
            freq = 1
        else:
            continue
        nfp_lower.append((noun.strip(), freq, polarity))  # .lower()

    stop_wordss = stopwords.words('english')
    nfp_model = []
    for (phrase, f, p) in nfp_lower:
        if phrase in stop_wordss:  # Phrases are nouns and pronouns. Stop words contain pronoun. "We love this.." vs "We hate this.. "
            continue
        if phrase in model:
            nfp_model.append((phrase, f, p))
        elif phrase.lower() in model:
            nfp_model.append((phrase.lower(), f, p))
        else:
            phrase_available = check_words_in_phrase_and_add_to_model(phrase, model)
            if phrase_available:
                nfp_model.append((phrase, f, p))

    nfp_terms = [phrase for (phrase, f, p) in nfp_model]
    return nfp_model, nfp_terms


def get_noun_freq_polarity(filename, nfp):
    # print(filename)
    f = codecs.open(filename, 'r', encoding="utf-8")
    for line in f:
        record = line.split("\t")
        nfp.append((record[0], 1, ast.literal_eval(record[-1])))


def get_model_and_terms(filename, nfp_model_terms_dict, model):
    if filename in nfp_model_terms_dict:
        noun_freq_polar_model, noun_freq_polar_terms = nfp_model_terms_dict[filename]
    else:
        noun_freq_polar = []
        get_noun_freq_polarity(filename, noun_freq_polar)
        noun_freq_polar_model, noun_freq_polar_terms = preprocess_words(noun_freq_polar, model)
        nfp_model_terms_dict[filename] = (noun_freq_polar_model, noun_freq_polar_terms)

    return noun_freq_polar_model, noun_freq_polar_terms


def get_term_vec(d, dictionary):
    filtered_d = []
    d_terms = []
    for i in range(0, len(d)):
        if d[i] > 0:
            d_terms.append(dictionary[i])
            filtered_d.append(i)
    return filtered_d, d_terms


def get_norm_freq_polarity(nfp, dictionary):
    # Normalizes the frequency of words with respect to it's document

    vocab_len = len(dictionary)
    d = zeros(vocab_len, dtype=double)
    # polarity_d = zeros(vocab_len, dtype=double)
    polarity_d = dict()
    neutral = 0

    for i in range(vocab_len):
        polarity_d[i] = neutral

    sum_freq = 0
    for idx, freq, pol in nfp:
        sum_freq = float(sum_freq) + float(freq)
    if sum_freq == 0:
        return d, polarity_d

    for (t1, freq, p) in nfp:
        if t1 in dictionary.token2id:
            i = dictionary.token2id[t1]
            # d[i] = float(freq) / float(sum_freq)
            d[i] = 1
            polarity_d[i] = p
    return d, polarity_d


def compute_semantic_distance_matrix(model, noun_freq_polar1_terms, noun_freq_polar2_terms, dictionary, filename1,
                                     filename2):
    # Dictionary is doc1 terms * doc2 terms and
    # distance matrix is ((doc1 terms + doc2 terms) * (doc1 terms + doc2 terms))
    # This dimension of matrix is required for Earth Mover distance computation

    vocab_len = len(dictionary)

    docset1 = set(noun_freq_polar1_terms)
    docset2 = set(noun_freq_polar2_terms)

    distance_matrix = np.full((vocab_len, vocab_len), 0.0)

    for i, t1 in dictionary.items():
        for j, t2 in dictionary.items():
            if t1 not in docset1 or t2 not in docset2:
                continue

            if t1 == t2 and model.strategy != "doc2vec":
                distance_matrix[i, j] = 0.00001
                continue

            distance_matrix[i, j] = model.compute_semantic_distance(t1, t2, "cosine")

    if np_sum(distance_matrix) == 0.0:
        print('The distance matrix is all zeros.')
        return None

    return distance_matrix


def non_matchings_penalty(filtered_doc1_idx, filtered_doc2_idx, matching_matrix, distance_matrix):
    num_no_matchings = 0

    for i in range(0, len(filtered_doc1_idx)):
        non_matched = True
        for j in range(0, len(filtered_doc2_idx)):
            flow = matching_matrix[filtered_doc1_idx[i]][filtered_doc2_idx[j]]
            dist = distance_matrix[filtered_doc1_idx[i]][filtered_doc2_idx[j]]
            if (flow > 0) and (dist < semantic_threshold):
                non_matched = False
        if non_matched:
            num_no_matchings += 1

    normalize_non_matches = (float(num_no_matchings) / (len(filtered_doc1_idx) + len(filtered_doc2_idx)))
    return normalize_non_matches


def compute_max_matching(normalized_freq1, normalized_freq2, dictionary, matching_matrix, distance_matrix):
    filtered_doc1_idx, doc1_terms = get_term_vec(normalized_freq1, dictionary)
    filtered_doc2_idx, doc2_terms = get_term_vec(normalized_freq2, dictionary)
    flow_matrix = zeros((len(doc1_terms), len(doc2_terms)))
    new_flow_matrix = zeros((len(doc1_terms), len(doc2_terms)))
    semantic_dist = zeros((len(doc1_terms), len(doc2_terms)))
    G = nx.Graph()

    for i in range(0, len(filtered_doc1_idx)):
        for j in range(0, len(filtered_doc2_idx)):

            flow_matrix[i, j] = matching_matrix[filtered_doc1_idx[i]][filtered_doc2_idx[j]]
            semantic_dist[i, j] = distance_matrix[filtered_doc1_idx[i]][filtered_doc2_idx[j]]

            if flow_matrix[i, j] == 0.0:
                continue

            if semantic_dist[i, j] > semantic_threshold:
                continue

            name1 = str(filtered_doc1_idx[i]) + "_0"
            name2 = str(filtered_doc2_idx[j]) + "_1"

            G.add_node(name1, bipartite=0)
            G.add_node(name2, bipartite=1)

            G.add_edge(name1, name2, weight=flow_matrix[i, j])

    mappings = nx.max_weight_matching(G)
    num_mappings = 0

    if len(mappings) == 0:
        return new_flow_matrix, num_mappings

    for key in mappings:
        vals_0 = key[0].split("_")
        vals_1 = key[1].split("_")

        if int(vals_0[-1]) == 0:
            i_val = int(vals_0[0])
            j_val = int(vals_1[0])
        else:
            i_val = int(vals_1[0])
            j_val = int(vals_0[0])

        i = filtered_doc1_idx.index(i_val)
        j = filtered_doc2_idx.index(j_val)
        num_mappings += 1

        new_flow_matrix[i, j] = G.get_edge_data(key[0], key[1])['weight']
        # print dictionary[filtered_doc1_idx[i]], dictionary[filtered_doc2_idx[j]]

    return new_flow_matrix, num_mappings


def compute_polarity_distance(normalized_freq1, normalized_freq2, dictionary, matching_matrix, distance_matrix,
                              pol1, pol2):
    filtered_doc1_idx, doc1_terms = get_term_vec(normalized_freq1, dictionary)
    filtered_doc2_idx, doc2_terms = get_term_vec(normalized_freq2, dictionary)
    flow_matrix = zeros((len(doc1_terms), len(doc2_terms)))
    num_mappings = 0

    semantic_dist = zeros((len(doc1_terms), len(doc2_terms)))
    polarity_distance = 0.0
    no_matchings_doc_level = True

    for i in range(0, len(filtered_doc1_idx)):
        for j in range(0, len(filtered_doc2_idx)):

            semantic_dist[i, j] = distance_matrix[filtered_doc1_idx[i]][filtered_doc2_idx[j]]
            flow_matrix[i, j] = matching_matrix[filtered_doc1_idx[i]][filtered_doc2_idx[j]]

            if semantic_dist[i, j] > semantic_threshold:
                continue

            if flow_matrix[i, j] == 0.0:
                continue

            num_mappings += 1
            no_matchings_doc_level = False

            one_pos = False
            two_pos = False
            if pol1[filtered_doc1_idx[i]] > 0.0:
                one_pos = True
            if pol2[filtered_doc2_idx[j]] > 0.0:
                two_pos = True

            if one_pos != two_pos:
                polarity_distance += 1

    if no_matchings_doc_level:
        polarity_distance = np.nan
    else:
        polarity_distance = polarity_distance / num_mappings

    return polarity_distance


def get_opinion_distance(model, noun_freq_polar1_model, noun_freq_polar1_terms, noun_freq_polar2_model,
                         noun_freq_polar2_terms, filename1, filename2):
    dictionary = Dictionary(documents=[noun_freq_polar1_terms, noun_freq_polar2_terms])

    # Compute the euclidean distance between word vectors
    semantic_distance_matrix = compute_semantic_distance_matrix(model, noun_freq_polar1_terms, noun_freq_polar2_terms,
                                                                dictionary, filename1, filename2)
    if semantic_distance_matrix is None:
        print("Semantic distance is none")
        return np.nan, np.nan

    # Get normalized frequency and it's polarity
    normalized_freq1, pol1 = get_norm_freq_polarity(noun_freq_polar1_model, dictionary)
    normalized_freq2, pol2 = get_norm_freq_polarity(noun_freq_polar2_model, dictionary)

    # Change output to d1_terms, d2_terms, d_matrix[len(d1_terms),len(d2_terms)]
    emd_distance, matching_matrix = emd_with_flow(normalized_freq1, normalized_freq2, semantic_distance_matrix)
    polarity_distance = compute_polarity_distance(normalized_freq1, normalized_freq2, dictionary, matching_matrix,
                                                  semantic_distance_matrix, pol1, pol2)

    return polarity_distance, emd_distance


def get_context_in_dictionary(term_context_dict, filename):
    fn = filename.replace("_values", "_context")

    fp = codecs.open(fn, encoding="utf-8")
    for line in fp.readlines():
        records = line.split("\t")
        term_context_dict[records[0].strip().lower()] = records[1].strip()


def compute_opinion_distance_among_all_files(filenames, result_file, model, args):
    nfp_model_terms_dict = {}
    undefined_pol_distance = []
    global semantic_threshold, embedding_strategy

    semantic_threshold = args.semantic_threshold
    embedding_strategy = args.embedding_strategy

    opinion_dist = np.zeros((len(filenames), len(filenames)))
    wmd_distances = np.zeros((len(filenames), len(filenames)))

    start_time = time.time()

    # Compute all pairs Opinion distance
    for index1, filename1 in enumerate(filenames):
        # Get the (noun, freq, polarity) in model and only noun phrase in terms
        noun_freq_polar1_model, noun_freq_polar1_terms = get_model_and_terms(filename1, nfp_model_terms_dict, model)

        for index2, filename2 in enumerate(filenames):
            if index1 == index2:
                opinion_dist[index1][index2] = 0.0
                wmd_distances[index1][index2] = 0.0
                continue

            if index1 > index2:
                continue

            noun_freq_polar2_model, noun_freq_polar2_terms = get_model_and_terms(filename2, nfp_model_terms_dict, model)
            pol_dist, emd_distance = get_opinion_distance(model, noun_freq_polar1_model, noun_freq_polar1_terms,
                                                          noun_freq_polar2_model, noun_freq_polar2_terms,
                                                          filename1, filename2)

            if pol_dist is np.nan or pol_dist is None:
                undefined_pol_distance.append((index1, index2))
            else:
                opinion_dist[index1][index2] = pol_dist
                wmd_distances[index1][index2] = emd_distance
                opinion_dist[index2][index1] = pol_dist
                wmd_distances[index2][index1] = emd_distance

    end_time = time.time()
    print("Opinion time", (end_time - start_time))
    opinion_dist = opinion_dist / np.nanmax(opinion_dist)

    for undefined in undefined_pol_distance:
        opinion_dist[undefined[0]][undefined[1]] = 1.0
        opinion_dist[undefined[1]][undefined[0]] = 1.0

    np.savetxt(result_file, opinion_dist, fmt='%.3f')

    return opinion_dist
