import torch
import glob
import logging
import random
import fnmatch
import ast
import yaml

# import dataset_lm.util as dsutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import gc
import os
import io

import nltk
nltk.download('punkt')
from transformers import BertTokenizer
from dataset.utils import tokenize_for_bleu_eval
from config.config import init_arg_parser
from asdl.grammar import GrammarRule, Grammar, ReduceAction
from asdl.ast_operation import *

from tqdm import tqdm

import json


QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?<!\\)(?P=quote)")

class APPSBaseDataset(torch.utils.data.Dataset):
    def __init__(self, path):

        # Do sanity checking
        with open(path) as f:
            fnames = json.load(f)

        self.problem_dirs = fnames  # Loaded from train/test split json files

        self.samples = []  # Should be set in initialize()

        self.initialize()


    def initialize(self):
        """
        Assume self.dataroot is set to folderName/data
        """

        all_samples = []
        skipped_problems = []

        print(f"Loading {len(self.problem_dirs)} problems from root.")

        for problem_name in tqdm(self.problem_dirs):

            question_fname = os.path.join('', problem_name, "question.txt")
            sols_fname = os.path.join('', problem_name, "solutions.json")
            starter_code = os.path.join('', problem_name, "starter_code.py")

            if os.path.exists(starter_code):
                answer_type = "\nUse Call-Based format\n"
            else:
                answer_type = "\nUse Standard Input format\n"

            if (not os.path.isfile(question_fname)) or (not os.path.isfile(sols_fname)):
                skipped_problems.append(problem_name)
                continue

            if (os.path.isfile(starter_code)):
                with open(starter_code, 'r') as f:
                    starter_code = f.read()
            else:
                starter_code = ""

            # Read the question description
            with open(question_fname, 'r') as f:
                question_str = f.read()

            # Read all the solutions
            with open(sols_fname, 'r') as f:
                sols_str_list = json.load(f)
                k = 0
                for sol_str in sols_str_list:
                    k += 1
                    # sol_str = reindent_code(sol_str)

                    sample = (question_str, starter_code, sol_str, answer_type)

                    all_samples.append(sample)

        print(f"Loaded {len(all_samples)} saamples from root.")
        print(f"Skipped {len(skipped_problems)} problems from root.")
        self.samples = all_samples

    def __len__(self):
        return len(self.samples)


def terminal_type(terminal):
    try:
        a = float(terminal)
    except (TypeError, ValueError, OverflowError):
        try:
            a = ast.literal_eval(terminal)
            return a
        except:
            return str(terminal)
    else:
        try:
            b = int(a)
        except (TypeError, ValueError, OverflowError):
            return a
        else:
            return b


def preprocess_data_apps(paths, act_dict, params):

    nl_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)

    loaded_examples = []

    for file_path, file_type in paths:

        nbr_unconstruct_example = 0

        for idx, example in enumerate(APPSBaseDataset(path=file_path).samples):
            try:

                query = example[0].strip()
                code = example[2].strip()

                if len(query) > 5000 or len(code) > 5000:
                    continue

                if params['model'] == 'bert':
                    query_tokens = nl_tokenizer.tokenize(query.lower())
                    query_tokens = ['[CLS]'] + query_tokens + ['[SEP]']
                else:
                    query_tokens = nltk.word_tokenize(query)

                if len(query_tokens) > params['len_max']:
                    nbr_unconstruct_example += 1
                    continue

                py_ast = ast.parse(code)

                actions, type, field, cardinality = ast2seq(py_ast, act_dict)

                encoded_actions = [
                    action[0].label if isinstance(action[0], GrammarRule) or isinstance(action[0], ReduceAction)
                    else (str(action[0]))
                    for action in actions]

                # TODO: check the problem with variable typing here
                # encoded_reconstr_actions = [*zip([act_dict[encoded_action] if encoded_action in act_dict
                #                                     else act_dict['Reduce'] if encoded_action == 'Reduce_primitif'
                # else (terminal_type(encoded_action)) for encoded_action in encoded_actions],
                #                                    [action[1] for action in actions])]
                #
                # code_reconstructed = seq2ast(make_iterlists(deque(encoded_reconstr_actions)))
                #
                # code_reconstructed = astor.to_source(code_reconstructed).rstrip()

                if len(encoded_actions) > params['len_max']:
                    nbr_unconstruct_example += 1
                else:
                    loaded_examples.append({'intent': query_tokens,
                                            'snippet_tokens': tokenize_for_bleu_eval(code.strip()),
                                            'snippet_actions': encoded_actions,
                                            'slot_map': {}})
            except:
                # print(code)
                nbr_unconstruct_example += 1

        print('first pass,' + file_type + ' examples processed %d' % (idx - nbr_unconstruct_example), file=sys.stderr)

        if file_type == 'train':
            data_set = loaded_examples
            data_set = pd.DataFrame(data_set)
            train_examples, dev_examples = train_test_split(data_set, test_size=0.1)
            train_examples.to_csv('dataset/data_apps/train.csv', index=False)
            dev_examples.to_csv('dataset/data_apps/dev.csv', index=False)
        else:
            test_examples = pd.DataFrame(loaded_examples)
            test_examples.to_csv('dataset/data_apps/test.csv', index=False)


if __name__ == '__main__':
    import json

    # bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #
    # dataset = APPSBaseDataset(
    #     path="dataset/data_apps/data_split/train.json"
    # )
    #
    # i = 10
    #
    # nl = dataset.samples[i][0]
    # starter_code = dataset.samples[i][1]
    # code = dataset.samples[i][2]
    # answer_type = dataset.samples[i][3]
    # code_ast = ast.parse(code)

    # print('Question: \n ', nl)
    # print('Code: \n', code)
    # print('Bert_intent :\n', len(bert_intent))

    args = init_arg_parser()
    params = yaml.load(open(args.config_file).read(), Loader=yaml.FullLoader)

    params = params['experiment_env']

    asdl_text = open('./asdl/grammar.txt').read()
    grammar, _, _ = Grammar.from_text(asdl_text)
    act_list = [GrammarRule(rule.constructor.name, rule.type.name, rule.fields) for rule in grammar]
    Reduce = ReduceAction('Reduce')
    act_dict = dict([(act.label, act) for act in act_list])
    act_dict[Reduce.label] = Reduce

    paths = [("dataset/data_apps/data_split/train.json", 'train'), ("dataset/data_apps/data_split/test.json", 'test')]

    preprocess_data_apps(paths, act_dict, params)


