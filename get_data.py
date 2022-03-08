import yaml

from components.vocabulary.create_vocab_dataset import create_vocab
from config.config import init_arg_parser
from dataset.data_conala.json_to_csv import data_creation
from dataset.data_conala.preprocess_conala import preprocess_data_conala
from dataset.data_django.preprocess_django import *
from dataset.data_github.preprocess_codesearchnet import preprocess_codesearchnet_dataset
from dataset.data_apps import apps_create_split
from dataset.data_apps.preprocess_apps import preprocess_data_apps

if __name__ == '__main__':
    args = init_arg_parser()
    params = yaml.load(open(args.config_file).read(), Loader=yaml.FullLoader)
    params = params['experiment_env']

    if params['dataset'] is 'conala' or 'codesearchnet' or 'apps':
        asdl_text = open('./asdl/grammar.txt').read()
    if params['dataset'] is 'django':
        asdl_text = open('./asdl/grammar2.txt').read()

    grammar, _, _ = Grammar.from_text(asdl_text)
    act_list = [GrammarRule(rule.constructor.name, rule.type.name, rule.fields) for rule in grammar]
    assert (len(grammar) == len(act_list))
    Reduce = ReduceAction('Reduce')
    act_dict = dict([(act.label, act) for act in act_list])
    act_dict[Reduce.label] = Reduce

    if params['dataset'] == 'conala':
        preprocess_data_conala(args.raw_path_conala, act_dict, params)
        data_creation(args.train_path_conala, args.raw_path_conala, params['number_merge_ex'], mode=params['mode'])
        data_creation(args.test_path_conala, args.raw_path_conala, params['number_merge_ex'], mode='test')

        if params['create_vocab'] == True:
            pydf_train = pd.read_csv(args.train_path_conala + 'conala-train.csv')
            pydf_valid = pd.read_csv(args.train_path_conala + 'conala-val.csv')

            pydf_vocabulary = pd.concat([pydf_train[['intent', 'snippet_actions']],
                                         pydf_valid[['intent', 'snippet_actions']]])

            create_vocab(pydf_train, act_dict, params)
            # pydf_train = pd.read_csv(args.train_path_conala + 'conala-train.csv')
            # create_vocab(pydf_train, act_dict, params)

    if params['dataset'] == 'django':
        Django.process_django_dataset(params, act_dict)

        if params['create_vocab'] == True:
            pydf_train = pd.read_csv(args.train_path_django + 'train.csv')
            pydf_valid = pd.read_csv(args.train_path_django + 'dev.csv')

            pydf_vocabulary = pd.concat([pydf_train[['intent', 'snippet_actions']],
                                         pydf_valid[['intent', 'snippet_actions']]])

            create_vocab(pydf_train, act_dict, params)
            # pydf_train = pd.read_csv('dataset/data_django/train.csv')
            # create_vocab(pydf_train, act_dict, params)

    if params['dataset'] == 'codesearchnet':

        print('codesearchnet preprocessing loading ...')

        preprocess_codesearchnet_dataset(args, params, act_dict, params['decode_max_time_step'])

        print()
        print('codesearchnet preprocessing done.')

        if params['create_vocab'] == True:
            print('codesearchnet vocabulary creation begins')
            pydf_train = pd.read_csv(args.train_path_codesearchnet + 'train.csv')
            pydf_valid = pd.read_csv(args.dev_path_codesearchnet + 'valid.csv')

            pydf_vocabulary = pd.concat([pydf_train[['intent', 'snippet_actions']],
                                         pydf_valid[['intent', 'snippet_actions']]])

            create_vocab(pydf_train, act_dict, params)

            # pydf_train = pd.read_csv(args.train_path_codesearchnet + 'train.csv')
            # create_vocab(pydf_train, act_dict, params)

            print()
            print('codesearchnet vocabulary creation done')

    if params['dataset'] == 'apps':

        print('apps preprocessing loading ...')

        paths = [(args.train_path_apps + 'data_split/train.json', 'train'), (args.test_path_apps + 'data_split/test.json', 'test')]

        apps_create_split

        preprocess_data_apps(paths, act_dict, params)

        print()
        print('apps preprocessing done.')

        if params['create_vocab'] == True:
            print('apps vocabulary creation begins')
            pydf_train = pd.read_csv(args.train_path_apps + 'train.csv')
            pydf_valid = pd.read_csv(args.train_path_apps + 'dev.csv')

            pydf_vocabulary = pd.concat([pydf_train[['intent', 'snippet_actions']],
                                         pydf_valid[['intent', 'snippet_actions']]])

            create_vocab(pydf_train, act_dict, params)

            print()
            print('apps vocabulary creation done')