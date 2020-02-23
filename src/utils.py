import os
import codecs
import numpy as np


def get_files_and_ground_truth(path, Aspect, docs_path="tagme_docs"):
    doc_path = path + docs_path + "/"
    files = os.listdir(doc_path)

    full_path_files = []
    groundtruth = []
    ground_truth = dict()

    gd_fp = codecs.open(path + "ground_truth.txt", encoding="utf-8")

    for line in gd_fp.readlines():
        records = line.split("\t")
        if records[-1].strip() not in Aspect:
            continue
        ground_truth[records[0].strip()] = int(records[1].strip())

    for file in files:
        if file in ground_truth:
            groundtruth.append(ground_truth[file])

    for file in files:
        if file in ground_truth:
            full_path_files.append(doc_path + file)

    return full_path_files, groundtruth


def create_docs(path, Aspect):
    doc_path = path + "docs" + "/"

    cluster_num = 0
    gd_fp = codecs.open(path + "ground_truth.txt", "w")
    ground_truth = dict()
    fp = codecs.open(path + "small.txt", encoding='iso-8859-1')

    doc = 0

    for line in fp.readlines():
        # print(line)
        records = line.split("\t")
        label = records[0].strip()
        if label == "":
            continue

        if label not in Aspect:
            continue

        fname = "doc_" + str(doc) + ".txt"
        fw = codecs.open(doc_path + fname, "w+")
        fw.write(" ".join(records[1:]))

        if label not in ground_truth:
            ground_truth[label] = cluster_num
        cluster_num += 1
        gd_fp.write(fname + "\t" + str(ground_truth[label]) + "\t" + label + "\n")
        doc += 1


def get_targets(path, topics, docs_path=None):
    if docs_path is None:
        from run_opinion_measure import docs_path

    global topic

    doc_path = path + docs_path + "/"

    files = os.listdir(doc_path)
    files = sorted(files)

    gd_fp = codecs.open(path + "ground_truth.txt", encoding="utf-8")
    targets = []
    ground_truth = dict()

    for line in gd_fp.readlines():
        records = line.split("\t")
        if int(records[2]) not in topics:
            continue
        ground_truth[str(records[0].strip()) + ".txt"] = records[-1].strip()

    for file in files:
        if file in ground_truth:
            targets.append(ground_truth[file])

    return targets


def reorder_distance_matrix(full_path_files, opinion_matrix_file, ground_truth):
    nfiles = []
    for file in full_path_files:
        nm = file.split("docs/")[-1]
        nfiles.append(nm)

    full_path_files = nfiles
    file_indices = dict()

    for index, file in enumerate(full_path_files):
        file_indices[index] = file

    matrix = read_distance_matrix_from_file(opinion_matrix_file, ignored_indices=list())
    reordered_matrix = np.zeros(matrix.shape)

    clusters = dict()
    for i, l in enumerate(ground_truth):
        l = int(l)
        if l not in clusters:
            clusters[l] = []
        clusters[l].append(i)

    from collections import OrderedDict
    renumber_indices = OrderedDict()
    renum = 0
    for key, vals in sorted(clusters.items()):
        for val in vals:
            renumber_indices[val] = renum
            renum += 1

    for key1, vals1 in sorted(clusters.items()):
        for key2, vals2 in clusters.items():
            for v1 in vals1:
                for v2 in vals2:
                    reordered_matrix[renumber_indices[v1]][renumber_indices[v2]] = matrix[v1][v2]

    gddd = []
    ff = []

    fw = codecs.open(opinion_matrix_file.replace(".txt", "_reordered.txt"), "w", encoding="utf-8")

    fw.write(",,")

    for key1, vals1 in sorted(clusters.items()):
        for v in vals1:
            fw.write(str(file_indices[v]) + ",")
            ff.append(str(file_indices[v]))

    fw.write("\n")
    fw.write(",,")

    for key1, vals1 in sorted(clusters.items()):
        for v in vals1:
            fw.write(str(key1) + ",")
            gddd.append(str(key1))

    fw.write("\n")

    index = 0
    for row in reordered_matrix:
        fw.write(ff[index] + "," + gddd[index] + ",")
        for col in row:
            fw.write(str(col) + ",")
        index += 1
        fw.write("\n")


def read_distance_matrix_from_file(filename, ignored_indices=list(), dimension=None):
    f = codecs.open(filename, 'r', encoding="utf-8")

    initialize = False
    X = ""

    if dimension == None:
        initialize = True
        dimension = 0
    else:
        X = np.zeros((dimension, dimension))

    i = 0
    index = 0

    for line in f:
        index += 1
        if index in ignored_indices:
            continue

        record = line.split()
        if initialize:
            dimension = len(record) - len(ignored_indices)
            X = np.zeros((dimension, dimension))
            initialize = False

        index_j = dimension + len(ignored_indices)

        j = 0
        for ij in range(index_j):
            if (ij + 1) in ignored_indices:
                continue

            X[i][j] = float(record[ij])

            j += 1
        i += 1

    return X
