import os
import codecs
import requests
import json
import nltk
import unicodedata
import math
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import numpy as np

nfp_path = "nfpv2"
docs_path = "tagme_docs"
sentiment_lexicon = "files/LEXICON_UG.txt"

tagme_spot_exceptions = ["don"]
per_file_noun_phrase_and_adjacent_words = dict()
per_file_noun_phrase_and_polarity = dict()
seperator_sign = "--SEPARATOR--"
aggregated_lexicon_dict = dict()
polarity_shifters = []
two_word_shifters = []
analyzer = SentimentIntensityAnalyzer()

tagme_exceptions = ["Games for May", "Voices of Animals and Men", "Aggression", "Stress (biology)", "John Favour",
                    "Internment Serial Number", "In A Good Way", "Prostitution law", "Production (economics)",
                    "The Substitute (Glee)", "Arthur Helps", "Sublimation (psychology)", "Vulnerability"]


def set_target_topic(name):
    global target

    final_title = ""
    text = re.sub(r'([^\s\w-]|_)+', '', name)
    for word in text.split():
        final_title += word + "_tagme_"

    target = final_title


def already_computed(file_name):
    file_name_store = file_name.replace("docs", nfp_path)
    file_name = file_name_store.replace(".txt", "")
    words = file_name + "_words.txt"
    values = file_name + "_values.txt"

    if os.path.exists(words) and os.path.exists(values):
        return True
    return False


def call_tagme(text):
    url = "https://tagme.d4science.org/tagme/tag"
    data = dict()
    data["lang"] = 'en'
    data["gcube-token"] = open("files/tagme_gcude_token.txt").read().strip()
    data["text"] = text
    data["long_text"] = 20
    data["epsilon"] = 0.4
    response = requests.post(url, data=data)
    return response.json()


def create_tagme_features(files, path):
    for file in files:
        # print(file)
        text = codecs.open(file, mode="r", encoding="utf-8").read()
        response = call_tagme(text)
        fname = file.split("/")[-1].replace(".txt", "") + ".json"
        with open(path + 'tagme_responses/' + fname, 'w') as outfile:
            json.dump(response, outfile)


def include_the_annotation(fixed_annotation):
    first = fixed_annotation[0]
    prob = first['link_probability']
    for ann in fixed_annotation[1:]:
        if prob < ann['link_probability']:
            return False
    return True


def get_sentences(max_annotations, sentences, read_str, start_index):
    for annotation in max_annotations:
        if "title" not in annotation:
            continue

        sentences.append(read_str[start_index:annotation["start"]])
        final_title = ""
        text = re.sub(r'([^\s\w-]|_)+', '', annotation["title"])
        for word in text.split():
            final_title += word + "_tagme_"

        final_title += " "
        sentences.append(final_title)
        start_index = annotation["end"]

    sentences.append(read_str[start_index:])
    # fw.write("".join(sentences))

    new_sentences = []
    for sentence in sentences:
        new_sent = sentence
        for annotation in max_annotations:
            if annotation['title'] in tagme_exceptions:
                continue
            text = annotation["spot"]
            if not text.isalnum():
                continue

            search = text + "\s"

            term = re.findall(search, sentence)
            if len(term) > 0:
                term = term[0].strip()
                final_title = ""
                text = re.sub(r'([^\s\w-]|_)+', '', annotation["title"])
                for word in text.split():
                    final_title += word + "_tagme_"

                final_title += " "
                new_sent = sentence.replace(term, final_title)

        new_sentences.append(new_sent)

    return new_sentences


def handle_no_annotation_case(response):
    max_annotations = []
    max_rho = 0
    max_index = 0
    for index, annotation in enumerate(response["annotations"]):
        if annotation['title'] in tagme_exceptions:
            continue

        if annotation['rho'] > max_rho:
            max_rho = annotation['rho']
            max_index = index
    max_annotations.append(response["annotations"][max_index])
    return max_annotations


def create_tagme_docs(files, path):
    link_prob = 0.05
    rho_th = 0.02

    for file in files:
        fname = file.split("/")[-1].replace(".txt", "") + ".json"
        # print(fname)

        response = json.load(codecs.open(path + 'tagme_responses/' + fname, 'r', 'utf-8-sig'))
        read_str = codecs.open(file, 'r', encoding='utf-8').read()
        fw = codecs.open(file.replace("docs", "tagme_docs"), 'w', encoding='utf-8')
        sentences = []

        start_index = 0
        max_annotations = []

        for index1, annotation1 in enumerate(response["annotations"]):
            start1 = annotation1["start"]
            end1 = annotation1["end"]

            if 'title' not in annotation1:
                continue

            if annotation1['link_probability'] < link_prob:
                continue

            if annotation1['rho'] < rho_th:
                continue

            if annotation1['title'] in tagme_exceptions:
                continue

            if annotation1['spot'] in tagme_spot_exceptions:
                continue

            fixed_annotation = []
            fixed_annotation.append(annotation1)
            requires_fix = False
            include_annotation = True

            for index2, annotation2 in enumerate(response["annotations"]):

                if index1 == index2:
                    continue

                if 'title' not in annotation2:
                    continue

                start2 = annotation2["start"]
                end2 = annotation2["end"]

                if annotation2['link_probability'] < link_prob:
                    continue

                if annotation1['rho'] < rho_th:
                    continue

                if annotation2['title'] in tagme_exceptions:
                    continue

                if annotation2['spot'] in tagme_spot_exceptions:
                    continue

                # Intersection.
                if (start1 <= end2) and (end1 >= start2):
                    requires_fix = True
                    include_annotation = False
                    fixed_annotation.append(annotation2)

            if requires_fix:
                include_annotation = include_the_annotation(fixed_annotation)

            if include_annotation:
                max_annotations.append(annotation1)

        if len(max_annotations) != 0:
            new_sentences = get_sentences(max_annotations, sentences, read_str, start_index)
            fw.write("".join(new_sentences))
        else:
            max_annotations = handle_no_annotation_case(response)
            new_sentences = get_sentences(max_annotations, sentences, read_str, start_index)
            fw.write("".join(new_sentences))


def extract_noun_phrase_and_its_adjacent_words(file_name):
    file = codecs.open(file_name, encoding="utf-8", mode="r")
    data = unicodedata.normalize("NFKD", file.read()).encode("ascii", "ignore").decode("utf-8")
    sent_text = nltk.sent_tokenize(data)

    for sentence in sent_text:
        for shifter in two_word_shifters:
            if shifter in sentence:
                sentence = sentence.replace(shifter, shifter.replace(" ", "_"))

        tokenized_words = nltk.word_tokenize(sentence, preserve_line=True)
        for index, word in enumerate(tokenized_words):
            if "_tagme_" in word:
                noun_phrase = word.replace("_tagme_", " ").strip()

                if file_name not in per_file_noun_phrase_and_adjacent_words:
                    per_file_noun_phrase_and_adjacent_words[file_name] = dict()
                    per_file_noun_phrase_and_polarity[file_name] = dict()

                if noun_phrase not in per_file_noun_phrase_and_adjacent_words[file_name]:
                    per_file_noun_phrase_and_adjacent_words[file_name][noun_phrase] = []
                    per_file_noun_phrase_and_polarity[file_name][noun_phrase] = dict()

                per_file_noun_phrase_and_adjacent_words[file_name][noun_phrase].append(seperator_sign)

                for adj_word in reversed(tokenized_words[0:index]):
                    if "_tagme_" in adj_word:
                        adj_word = adj_word.replace("_tagme_", " ").strip().split()
                    else:
                        adj_word = [adj_word.strip()]
                    per_file_noun_phrase_and_adjacent_words[file_name][noun_phrase].extend(adj_word)

                per_file_noun_phrase_and_adjacent_words[file_name][noun_phrase].append(seperator_sign)

                for adj_word in tokenized_words[index + 1:]:
                    if "_tagme_" in adj_word:
                        adj_word = adj_word.replace("_tagme_", " ").strip().split()
                    else:
                        adj_word = [adj_word.strip()]
                    per_file_noun_phrase_and_adjacent_words[file_name][noun_phrase].extend(adj_word)


def get_average_polarity(sentiment_score, file_name, noun_phrase, distance_vector):
    pos = 0.0
    neg = 0.0
    non_zeros = True
    nnonzeros = 0

    for l in sentiment_score:
        if l > 0.0:
            pos += l
            non_zeros = False
            nnonzeros += 1
        if l < 0.0:
            neg += l
            non_zeros = False
            nnonzeros += 1

    if non_zeros:
        aggregated_pol = 0.0
    else:
        neg = -1 * neg
        aggregated_pol = (pos - neg) / (pos + neg + 1)

    num_polarity = 0
    shift = 1
    for index, word in enumerate(per_file_noun_phrase_and_adjacent_words[file_name][noun_phrase]):
        if word.lower().strip() in polarity_shifters:
            num_polarity += 1
            shift = math.pow(-1, num_polarity)
    return shift * aggregated_pol


def assign_polarity_to_adjacent_words(file_name):
    lemmatizer = WordNetLemmatizer()

    for noun_phrase in per_file_noun_phrase_and_adjacent_words[file_name]:
        sentiment_score = []
        sentiment_words = []
        distance_vector = []
        dist = 1
        for word in per_file_noun_phrase_and_adjacent_words[file_name][noun_phrase]:
            if seperator_sign in word:
                dist = 1
            distance_vector.append(dist)
            dist += 1

        for dist_index, word in enumerate(per_file_noun_phrase_and_adjacent_words[file_name][noun_phrase]):
            if seperator_sign in word:
                continue

            word = word.lower()
            reweight = math.pow(distance_vector[dist_index], -0.5)

            if word not in aggregated_lexicon_dict:
                word = lemmatizer.lemmatize(word, pos=wn.NOUN)
                if word.replace("-", "") in aggregated_lexicon_dict:
                    word = word.replace("-", "")

            if word in aggregated_lexicon_dict:
                val = aggregated_lexicon_dict[word]
                if val != 0.0:
                    sentiment_score.append(val * reweight)
                    sentiment_words.append(word)

        pol_val = get_average_polarity(sentiment_score, file_name, noun_phrase, distance_vector)
        per_file_noun_phrase_and_polarity[file_name][noun_phrase] = pol_val


def read_lexicon():
    fp = codecs.open(sentiment_lexicon, "r", encoding="utf-8")
    for line in fp.readlines():
        polar = line.split()
        aggregated_lexicon_dict[polar[0].strip().lower()] = float(polar[-1].strip())


def write_details(fp, stored_dict):
    for noun_phrase in stored_dict:

        fp.write(noun_phrase)
        if type(stored_dict[noun_phrase]) == float:
            fp.write("\t")
            fp.write(str(stored_dict[noun_phrase]))
        elif isinstance(stored_dict[noun_phrase][0], np.float64):
            fp.write("\t")
            fp.write(str(stored_dict[noun_phrase]))
        else:
            for word in stored_dict[noun_phrase]:
                if seperator_sign in word:
                    continue
                fp.write("\t")
                fp.write(word)
        fp.write("\n")


def write_extract_np_and_its_polarity(file_name):
    file_name_store = file_name.replace("tagme_docs", nfp_path)
    store_fp_words = codecs.open(file_name_store.replace(".txt", "") + "_words.txt", "w")
    store_fp_values = codecs.open(file_name_store.replace(".txt", "") + "_values.txt", "w")
    write_details(store_fp_words, per_file_noun_phrase_and_adjacent_words[file_name])
    write_details(store_fp_values, per_file_noun_phrase_and_polarity[file_name])


def read_polarity_shifters():
    global polarity_shifters, two_word_shifters
    polarity_shifters.extend(
        ['no', 'not', 'inconclusive', 'excluded', "n't", "incompatible", "prevent", 'exacerbate'
            , 'without', 'little_act', 'cannot', 'limit', 'outweigh', 'unless', 'reduce', 'get_even',
         'less', 'rarely', 'negation', 'none', 'displaced',
         'higher_than' 'relocation', 'dislocation', 'relocate', 'resettled', 're-housed'])
    two_word_shifters.extend(['little act', 'get even', 'higher than'])


def represent_opinion_in_nfp(files, path):
    create_tagme_features(files, path)
    create_tagme_docs(files, path)
    read_lexicon()
    read_polarity_shifters()
    for file_name in files:
        file_name = file_name.replace("docs", "tagme_docs")
        extract_noun_phrase_and_its_adjacent_words(file_name)
        assign_polarity_to_adjacent_words(file_name)
        write_extract_np_and_its_polarity(file_name)
