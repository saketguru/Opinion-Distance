from baselines import tf_idf_baseline, wmd_baseline, doc2vec_baseline, bert_baselines
from get_tagme_opinion_repr import represent_opinion_in_nfp
import codecs
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import ast
import time
from utils import get_files_and_ground_truth, reorder_distance_matrix
from Model import Model
from compute_opinion_distance import compute_opinion_distance_among_all_files
from eval import get_evaluation_scores


def get_opinion_distance(filename, filepath, nfp_available):
    print("Finding opinion distance", filename)

    nfps_path = "nfpv2"

    if not nfp_available:
        start_time = time.time()
        represent_opinion_in_nfp(full_path_files, args.path)
        end_time = time.time()
        print("Representation time ", (end_time - start_time))

    model = None
    if args.embedding_strategy == 'word2vec':
        model = Model("word2vec")
    elif args.embedding_strategy == 'doc2vec':
        model = Model("doc2vec")

    files = []
    for file in full_path_files:
        full_name = file.replace("docs", nfps_path).replace(".txt", "_values.txt")
        files.append(full_name)

    X = compute_opinion_distance_among_all_files(files, filename, model, args)

    get_evaluation_scores(X=X, ground_truth_labels=ground_truth, clusters=len(set(ground_truth)), filepath=filepath,
                          result_file=None, files=files)

    filepath.write("\n")
    filepath.flush()


def tf_idf_result(result_file):
    print("\n ------------\n")
    print("Finding tfidf distance")
    X = tf_idf_baseline(full_path_files, result_file, 'filename')
    reorder_distance_matrix(full_path_files, result_file, ground_truth)
    filepath = open(result_file.replace("dist_mats", "results"), "w+")
    get_evaluation_scores(X, ground_truth, len(set(ground_truth)), filepath)
    filepath.write("\n")
    filepath.flush()


def wmd_result(model, result_file):
    print("\n ------------\n")
    print("Finding wmd distance")
    X = None
    if X is None:
        X = wmd_baseline(full_path_files, result_file, model)
    reorder_distance_matrix(full_path_files, result_file, ground_truth)
    filepath = open(result_file.replace("dist_mats", "results"), "w+")
    get_evaluation_scores(X, ground_truth, len(set(ground_truth)), filepath)
    filepath.write("\n")
    filepath.flush()
    return X


def doc2vec_result(model, result_file):
    print("\n ------------\n")
    print("Finding doc2vec distance")
    X = doc2vec_baseline(full_path_files, result_file, model)
    reorder_distance_matrix(full_path_files, result_file, ground_truth)
    result_file = result_file.replace("dist_mats", "results")
    filepath = open(result_file.replace("dist_mats", "results"), "w+")
    get_evaluation_scores(X, ground_truth, len(set(ground_truth)), filepath)
    filepath.write("\n")
    filepath.flush()


def bert_result(result_file):
    print("\n ------------\n")
    print("Finding BERT distance")
    X = bert_baselines(full_path_files, result_file)
    reorder_distance_matrix(full_path_files, result_file, ground_truth)
    result_file = result_file.replace("dist_mats", "results")
    filepath = open(result_file.replace("dist_mats", "results"), "w+")
    get_evaluation_scores(X, ground_truth, len(set(ground_truth)), filepath)
    filepath.write("\n")
    filepath.flush()


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--path', required=True)
    parser.add_argument('--embedding_strategy', required=False, default='word2vec',
                        choices=['word2vec'], help='Semantic Distance Model')
    parser.add_argument('--baselines', default=False, type=ast.literal_eval)
    parser.add_argument('--semantic_threshold', required=False, default=0.6, type=float)
    #####################################
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    docs_path = "tagme_docs"
    full_path_files, ground_truth = None, None

    if "CiviQ_Seanad" in args.path:
        Aspect = ["Abolish", "Do not Abolish", "Democracy"]
        full_path_files, ground_truth = get_files_and_ground_truth(args.path, Aspect, docs_path="docs")

    filename = ("%s/dist_mats/OD_embedding_strategy_%s.txt") % (args.path, args.embedding_strategy)
    fp = codecs.open(filename.replace("dist_mats", "results"), "w")

    if args.baselines:
        tf_idf_result(args.path + "dist_mats/tf_idf.txt")
        # wmd_dist = wmd_result(Model("word2vec"), args.path + "dist_mats/wmd.txt")
        # bert_result(args.path + "dist_mats/bert.txt")
        # doc2vec_result(Model("doc2vec"), args.path + "dist_mats/doc2vec.txt")

    get_opinion_distance(filename=filename, filepath=fp, nfp_available=False)
