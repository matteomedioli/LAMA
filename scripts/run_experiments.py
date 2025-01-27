# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from batch_eval_KB_completion import main as run_evaluation
from batch_eval_KB_completion import load_file
from lama.modules import build_model_by_name
import pprint
import statistics
from collections import defaultdict
import json
from tqdm import tqdm

def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    input_param={
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    },
    use_negated_probes=False,
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_Precision1 = []
    all_Precision10 = []
    all_mrr = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    results_file = open("last_results.csv", "w+")

    for relation in relations:
        pp.pprint(relation)
        PARAMETERS = {
            "dataset_filename": "{}{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "common_vocab_filename": "pre-trained_language_models/common_vocab_lowercased.txt",
            "template": "",
            "bert_vocab_name": "vocab.txt",
            "batch_size": 32,
            "logdir": "output",
            "full_logdir": "output/results/{}/{}".format(
                input_param["label"], relation["relation"]
            ),
            "lowercase": True,
            "max_sentence_length": 100,
            "threads": -1,
            "interactive": False,
            "use_negated_probes": use_negated_probes,
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]
            if use_negated_probes:
                PARAMETERS["template_negated"] = relation["template_negated"]

        PARAMETERS.update(input_param)
        print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)

        # see if file exists
        try:
            data = load_file(args.dataset_filename)
        except Exception as e:
            print("Relation {} excluded.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue

        if model is None:
            [model_type_name] = args.models_names
            model = build_model_by_name(model_type_name, args)

        Precision1, Precision10, MRR = run_evaluation(args, shuffle_data=False, model=model)
        print("P@1 : {}".format(Precision1), flush=True)
        all_Precision1.append(Precision1)
        all_Precision10.append(Precision10)
        all_mrr.append(MRR)

        results_file.write(
            "{},{}\n".format(relation["relation"], round(Precision1 * 100, 2))
        )
        results_file.write(
            "{},{}\n".format(relation["relation"], round(Precision10 * 100, 2))
        )
        results_file.write(
            "{},{}\n".format(relation["relation"], round(MRR* 100, 2))
        )
        results_file.flush()

        if "type" in relation:
            type_Precision1[relation["type"]].append(Precision1)
            data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(len(data))

    mean_p1 = statistics.mean(all_Precision1)
    mean_p10 = statistics.mean(all_Precision10)
    mean_MRR = statistics.mean(all_mrr)
    print("@@@ {} - mean P@1: {}".format(input_param["label"], mean_p1))
    print("@@@ {} - mean P@10: {}".format(input_param["label"], mean_p10))
    print("@@@ {} - mean MRR: {}".format(input_param["label"], mean_MRR))
    results_file.close()

    for t, l in type_Precision1.items():

        print(
            "@@@ ",
            input_param["label"],
            t,
            statistics.mean(l),
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )

    return mean_p1, mean_p10, mean_MRR


def get_TREx_parameters(data_path_pre="data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_GoogleRE_parameters():
    relations = [
        {
            "relation": "place_of_birth",
            "template": "[X] was born in [Y] .",
            "template_negated": "[X] was not born in [Y] .",
        },
        {
            "relation": "date_of_birth",
            "template": "[X] (born [Y]).",
            "template_negated": "[X] (not born [Y]).",
        },
        {
            "relation": "place_of_death",
            "template": "[X] died in [Y] .",
            "template_negated": "[X] did not die in [Y] .",
        },
    ]
    data_path_pre = "data/Google_RE/"
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post


def get_ConceptNet_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "ConceptNet/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_Squad_parameters(data_path_pre="/ddn/medioli/lama/data/"): #/data/medioli/lama/data/
    relations = [{"relation": "test"}]
    data_path_pre += "Squad/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def run_all_LMs(parameters):
    results={"baseline":{"mrr":[], "p10":[], "p1":[]},
            "wordnet":{"mrr":[], "p10":[], "p1":[]},
            "freebase":{"mrr":[], "p10":[], "p1":[]}}
    for i in tqdm(range(10000, 800000, 10000)):
        LMs = [
            #{
            #    # HuggingFace Baseline
            #    "lm": "bert",
            #    "label": "bert_base_uncased",
            #    "models_names": ["bert"],
            #    "bert_model_name": "bert-base-uncased",
            #    "bert_model_dir": None,
            #    "common_vocab_filename": "pre-trained_language_models/common_vocab_lowercased.txt",
            #    "bert_vocab_name": "vocab.txt"
            #},
            {
                "lm": "bert-custom-baseline",
                "label": "bert_custom_baseline",
                "models_names": ["bert"],
                "bert_model_name": "baseline",
                "bert_model_dir": "/data/medioli/models/mlm/bert_wikipedia_5_BASELINE_WITH_GAT_OCTOBER/checkpoint-",
                "common_vocab_filename": "pre-trained_language_models/common_vocab_lowercased.txt",
                "bert_vocab_name": "vocab.txt",
            },
            {
                "lm": "bert-custom-regularized",
                "label": "bert_custom_regularized",
                "models_names": ["bert"],
                "bert_model_name": "wordnet",
                "bert_model_dir": "/data/medioli/models/mlm/bert_wikipedia_5_SECOND_TEST_WITH_GAT_FIX_64/checkpoint-",
                "common_vocab_filename": "pre-trained_language_models/common_vocab_lowercased.txt",
                "bert_vocab_name": "vocab.txt"
            },
            {
                "lm": "bert-custom-regularized",
                "label": "bert_custom_regularized",
                "models_names": ["bert"],
                "bert_model_name": "freebase",
                "bert_model_dir": "/ddn/medioli/models/mlm/bert_wikipedia_5_FREEBASE_GENNAIO_freebase/checkpoint-",
                "common_vocab_filename": "pre-trained_language_models/common_vocab_lowercased.txt",
                "bert_vocab_name": "vocab.txt"
            }
        ]
        for lm in LMs:
            lm["bert_model_dir"] = lm["bert_model_dir"]+str(i)
            print(lm["label"])
            p1, p10, mrr = run_experiments(*parameters, input_param=lm, use_negated_probes=False)
            results[lm["bert_model_name"]]["mrr"].append(mrr)
            results[lm["bert_model_name"]]["p10"].append(p10)
            results[lm["bert_model_name"]]["p1"].append(p1)

    with open('results_febbraio.json', 'w') as fp:
        json.dump(results, fp)


if __name__ == "__main__":

    # print("1. Google-RE")
    # parameters = get_GoogleRE_parameters()
    # run_all_LMs(parameters)

    #print("2. T-REx")
    #parameters = get_TREx_parameters()
    #run_all_LMs(parameters)

    # print("3. ConceptNet")
    # parameters = get_ConceptNet_parameters()
    # run_all_LMs(parameters)

    print("4. SQuAD")
    parameters = get_Squad_parameters()
    run_all_LMs(parameters)




'''
LMs = [
        {
            # HuggingFace Baseline
            "lm": "bert",
            "label": "bert_base_uncased",
            "models_names": ["bert"],
            "bert_model_name": "bert-base-uncased",
            "bert_model_dir": None,
            "common_vocab_filename": "pre-trained_language_models/common_vocab_lowercased.txt",
            "bert_vocab_name": "vocab.txt"
        },
        {
            "lm": "bert-custom-baseline",
            "label": "bert_custom_baseline",
            "models_names": ["bert"],
            "bert_model_name": "bert-custom-baseline",
            "bert_model_dir": "/home/med/Scrivania/DATA-SERVER/language models/bert_baseline_815000",
            "common_vocab_filename": "pre-trained_language_models/common_vocab_lowercased.txt",
            "bert_vocab_name": "vocab.txt",
        },
        {
            "lm": "bert-custom-regularized",
            "label": "bert_custom_regularized",
            "models_names": ["bert"],
            "bert_model_name": "bert-custom-regularized",
            "bert_model_dir": "/home/med/Scrivania/DATA-SERVER/language models/bert_wordnet_815000",
            "common_vocab_filename": "pre-trained_language_models/common_vocab_lowercased.txt",
            "bert_vocab_name": "vocab.txt"
        },
        {
            "lm": "bert-custom-regularized",
            "label": "bert_custom_regularized",
            "models_names": ["bert"],
            "bert_model_name": "bert-custom-regularized",
            "bert_model_dir": "/home/med/Scrivania/DATA-SERVER/language models/bert_freebase_815000",
            "common_vocab_filename": "pre-trained_language_models/common_vocab_lowercased.txt",
            "bert_vocab_name": "vocab.txt"
        }
    ]
MEN_top10 = [
    ("sun", "sunlight"),
    ("automobile", "car"),
    ("river", "water"),
    ("stairs", "staircase"),
    ("morning", "sunrise"),
    ("rain", "storm"),
    ("cat", "kittens"),
    ("dance", "dancers"),
    ("camera", "photography"),
    ("cat", "feline")]

MEN_lowest10 = [
    ("posted", "tulip"),
    ("grave", "hat"),
    ("apple", "cute"),
    ("angel", "gasoline"),
    ("giraffe", "harbor"),
    ("feathers", "truck"),
    ("festival", "whiskers"),
    ("muscle", "tulip"),
    ("bikini", "pizza"),
    ("bakery", "zebra")]

for lm in LMs:
    bert = load_model(lm)
    ------------------------------------------------------------
    def load_model(
        input_param={
            "lm": "bert",
            "label": "bert_large",
            "models_names": ["bert"],
            "bert_model_name": "bert-large-cased",
            "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
        },
):
    model = None

    PARAMETERS = {
        "dataset_filename": "{}{}{}".format(
            data_path_pre, relation["relation"], data_path_post
        ),
        "common_vocab_filename": "pre-trained_language_models/common_vocab_lowercased.txt",
        "template": "",
        "bert_vocab_name": "vocab.txt",
        "batch_size": 32,
        "logdir": "output",
        "full_logdir": "output/results/{}/{}".format(
            input_param["label"], relation["relation"]
        ),
        "lowercase": True,
        "max_sentence_length": 100,
        "threads": -1,
        "interactive": False,
        "use_negated_probes": use_negated_probes,
    }
    
    PARAMETERS = {}
    PARAMETERS.update(input_param)
    print(PARAMETERS)

    args = argparse.Namespace(**PARAMETERS)

    if model is None:
        [model_type_name] = args.models_names
        model = build_model_by_name(model_type_name, args)
    return model
'''