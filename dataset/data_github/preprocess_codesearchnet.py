import re
from pathlib import Path

import nltk
import pandas as pd
from docstring_parser import parse
from transformers import BertTokenizer

nltk.download('punkt')

from asdl.ast_operation import *

QUOTED_TOKEN_RE = re.compile(r"(?P<quote>''|[`'\"])(?P<string>.*?)(?P=quote)")


def preprocess_codesearchnet_dataset(args, params, act_dict, max_len):
    """
    data processing from CodeSearchNet raw token
    """

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokenizer.add_tokens(['var_0', 'str_0', 'var_1', 'str_1', 'var_2', 'str_2'])

    codesearchnet_columns = ['docstring', 'code', 'code_tokens']

    python_files = sorted(Path(args.raw_path_codesearchnet).glob('**/*.gz'))

    pydf = jsonl_list_to_dataframe(python_files, codesearchnet_columns)

    print('number of examples before preprocessing:', len(pydf))

    pydf = pydf[pydf['docstring'].map(len) < max_len]

    pydf['code'] = pydf.code.apply(if_class_method)

    pydf = pydf.dropna()

    pydf['code'] = pydf.code.apply(remove_docstring)

    pydf['code'] = pydf.apply(lambda x: compare_length(x['code'], x['docstring'], 60), axis=1)

    pydf = pydf.dropna()

    # pydf.to_csv(args.train_path_codesearchnet + 'demonstrate.csv', index=False)

    pydf['docstring'] = pydf.docstring.apply(multiple_replace)

    pydf['docstring'], pydf['slot_map'] = zip(*pydf['docstring'].map(canonicalize_intent))

    pydf['code'] = pydf.apply(lambda x: canonicalize_code(x['code'], x['slot_map']), axis=1)

    pydf['py_ast'] = pydf.code.apply(get_ast)

    pydf = pydf.dropna()

    values = pydf.apply(
        lambda x: ast2seq(x['py_ast'], action_dict=act_dict, parent_type=(), parent_field=(), parent_cardinality=()),
        axis=1)

    pydf['actions'], pydf['parent_types'], pydf['parent_fields'], pydf['parent_cardinalities'] = np.array(
        values.to_list()).T

    pydf = pydf[pydf['actions'].map(len) < max_len]

    pydf['snippet_actions'] = pydf.actions.apply(get_actions)

    pydf['snippet_actions'].to_csv(args.train_path_codesearchnet + 'actions.csv', index=False)

    pydf['docstring'].to_csv(args.train_path_codesearchnet + 'dataset.csv', index=False)

    pydf = pydf.rename(columns={"docstring": "intent"})

    pydf = pydf.drop(['py_ast', 'actions', 'code'], 1)

    print('number of examples after preprocessing:', len(pydf))

    pydf = pydf.rename(columns={"code_tokens": "snippet_tokens"})

    pydf_train, pydf_valid, pydf_test = np.split(pydf.sample(frac=1, random_state=42),
                                                 [int(.8 * len(pydf)), int(.9 * len(pydf))])

    print('train_set size', len(pydf_train))

    pydf_train.to_csv(args.train_path_codesearchnet + 'train.csv', index=False)

    print('dev_set size', len(pydf_valid))

    pydf_valid.to_csv(args.dev_path_codesearchnet + 'valid.csv', index=False)

    print('test_size size', len(pydf_valid))

    pydf_test.to_csv(args.test_path_codesearchnet + 'test.csv', index=False)


def canonicalize_intent(intent, mode='seq2seq', tokenizer='None'):
    # handle the following special case: quote is `''`
    marked_token_matches = QUOTED_TOKEN_RE.findall(intent)

    slot_map = dict()
    var_id = 0
    str_id = 0
    for match in marked_token_matches:
        quote = match[0]
        value = match[1]
        quoted_value = quote + value + quote

        # try:
        #     # if it's a number, then keep it and leave it to the copy mechanism
        #     float(value)
        #     intent = intent.replace(quoted_value, value)
        #     continue
        # except:
        #     pass

        slot_type = infer_slot_type(quote, value)

        if slot_type == 'var':
            slot_name = 'var_%d' % var_id
            var_id += 1
            slot_type = 'var'
        else:
            slot_name = 'str_%d' % str_id
            str_id += 1
            slot_type = 'str'

        # slot_id = len(slot_map)
        # slot_name = 'slot_%d' % slot_id
        # # make sure slot_name is also unicode
        # slot_name = unicode(slot_name)

        intent = intent.replace(quoted_value, slot_name)

        slot_map[slot_name] = {'value': value.strip().encode().decode('unicode_escape', 'ignore'),
                               'quote': quote,
                               'type': slot_type}
    if mode == 'bert':
        intent = tokenizer.tokenize(intent.lower())
        intent = ['[CLS]'] + intent + ['[SEP]']
        return intent, slot_map
    else:
        return nltk.word_tokenize(intent.lower()), slot_map


def infer_slot_type(quote, value):
    if quote == '`' and value.isidentifier():
        return 'var'
    return 'str'


def replace_identifiers_in_ast(py_ast, identifier2slot):
    for node in ast.walk(py_ast):
        for k, v in list(vars(node).items()):
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            # Python 3
            # if isinstance(v, str) or isinstance(v, unicode):
            if isinstance(v, str):
                if v in identifier2slot:
                    slot_name = identifier2slot[v]
                    # Python 3
                    # if isinstance(slot_name, unicode):
                    #     try: slot_name = slot_name.encode('ascii')
                    #     except: pass

                    setattr(node, k, slot_name
                            )


def canonicalize_code(code, slot_map):
    try:
        string2slot = {x['value']: slot_name for slot_name, x in list(slot_map.items())}
        py_ast = ast.parse(code)
        replace_identifiers_in_ast(py_ast, string2slot)
        canonical_code = astor.to_source(py_ast).strip()

        # the following code handles the special case that
        # a list/dict/set mentioned in the intent, like
        # Intent: zip two lists `[1, 2]` and `[3, 4]` into a list of two tuples containing elements at the same index in each list
        # Code: zip([1, 2], [3, 4])

        entries_that_are_lists = [slot_name for slot_name, val in slot_map.items() if
                                  is_enumerable_str(val['value'])]
        if entries_that_are_lists:
            for slot_name in entries_that_are_lists:
                list_repr = slot_map[slot_name]['value']
                # if list_repr[0] == '[' and list_repr[-1] == ']':
                first_token = list_repr[0]  # e.g. `[`
                last_token = list_repr[-1]  # e.g., `]`
                fake_list = first_token + slot_name + last_token
                slot_map[fake_list] = slot_map[slot_name]
                # else:
                #     fake_list = slot_name

                canonical_code = canonical_code.replace(list_repr, fake_list)

        return canonical_code
    except:
        return np.nan


def decanonicalize_code(code, slot_map):
    for slot_name, slot_val in slot_map.items():
        if is_enumerable_str(slot_name):
            code = code.replace(slot_name, slot_val['value'])

    slot2string = {x[0]: x[1]['value'] for x in list(slot_map.items())}
    py_ast = ast.parse(code)
    replace_identifiers_in_ast(py_ast, slot2string)
    raw_code = astor.to_source(py_ast).strip()
    # for slot_name, slot_info in slot_map.items():
    #     raw_code = raw_code.replace(slot_name, slot_info['value'])

    return raw_code


def is_enumerable_str(identifier_value):
    """
    Test if the quoted identifier value is a list
    """

    return len(identifier_value) > 2 and identifier_value[0] in ('{', '(', '[') and identifier_value[-1] in (
        '}', ']', ')')


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


def jsonl_list_to_dataframe(file_list, columns):
    """Load a list of jsonl.gz files into a pandas DataFrame."""
    trainset = pd.concat([pd.read_json(f,
                                       orient='records',
                                       compression='gzip',
                                       lines=True)[columns]
                          for f in file_list], sort=False)
    return trainset


def get_ast(code):
    try:
        return ast.parse(code)
    except:
        return np.nan


def remove_docstring(code):
    try:
        py_ast = ast.parse(code)
        for node in ast.walk(py_ast):
            # let's work only on functions & classes definitions
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                continue

            if not len(node.body):
                continue

            if not isinstance(node.body[0], ast.Expr):
                continue

            if not hasattr(node.body[0], 'value') or not isinstance(node.body[0].value, ast.Str):
                continue

            node.body = node.body[1:]
        return astor.to_source(py_ast)
    except:
        return np.nan


def get_actions(seq_action):
    encoded_actions = [
        action[0].label if isinstance(action[0], GrammarRule)
                           or isinstance(action[0], ReduceAction)
        else (str(action[0]))
        for action in seq_action
    ]
    return encoded_actions


def multiple_replace(doc_str):
    replacement = {}
    try:
        doc_str_parsed = parse(doc_str)
        params = [param.arg_name for param in doc_str_parsed.params]
        for param in params:
            replacement[param] = '`' + param + '`'
        if replacement == {}:
            return doc_str
        else:
            # Create a regular expression  from the dictionary keys
            regex = re.compile("(%s)" % "|".join(map(re.escape, replacement.keys())))
            # For each match, look-up corresponding value in dictionary
            return regex.sub(lambda mo: replacement[mo.string[mo.start():mo.end()]], doc_str)
    except:
        return doc_str


def compare_length(code_string, doc_string, window):
    try:
        docstring_size = len(doc_string)
        code_string_size = len(code_string)
        if docstring_size - window <= code_string_size <= docstring_size + window:
            return code_string
        else:
            return np.nan
    except:
        return np.nan


def if_class_method(code):
    try:
        if 'self.' in code:
            return np.nan
        else:
            return code
    except:
        return np.nan