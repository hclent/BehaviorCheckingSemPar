"""
Load raw or processed data.
"""
import collections
import json
import os
import pickle
import shutil

from src.data_processor.data_utils import WIKISQL, SPIDER, OTHERS
from src.data_processor.data_utils import Text2SQLExample, AugmentedText2SQLExample
from src.data_processor.path_utils import get_norm_tag, get_data_augmentation_tag
from src.data_processor.path_utils import get_processed_data_path, get_vocab_path
from src.data_processor.schema_loader import load_schema_graphs_spider, load_schema_graphs_wikisql
from src.data_processor.sql.sql_reserved_tokens import sql_reserved_tokens, sql_reserved_tokens_revtok
from src.data_processor.vocab_utils import is_functional_token, Vocabulary, value_vocab
import src.utils.utils as utils


def load_processed_data(args):
    """
    Load preprocessed data file.
    """
    if args.process_sql_in_execution_order:
        if "scfg" in args.dataset_name or "nlperturb" in args.dataset_name:
            pred_restored_cache_path = os.path.join(
                "model/scfg_spider.sqlova.ppl.2.dn.eo.feat.bert-base-uncased.xavier-768-512", '{}.eo.pred.restored.pkl'.format(args.dataset_name))
            print(f"SCFG pred_restored_cache_path: {pred_restored_cache_path}")
        else:
            split = 'test' if args.test else 'dev'
            pred_restored_cache_path = os.path.join(
                args.model_dir, '{}.eo.pred.restored.pkl'.format(split))
            if not os.path.exists(pred_restored_cache_path):
                print(f"`this doesnt exiiiiiiiiisstttttt")
                cache_path = os.path.join(args.data_dir, '{}.eo.pred.restored.pkl'.format(split))
                shutil.copyfile(cache_path, pred_restored_cache_path)
                print('execution order restoration cache copied')
                print('source: {}'.format(cache_path))
                print('dest: {}'.format(pred_restored_cache_path))
                print()

    in_pkl = get_processed_data_path(args)
    print('loading preprocessed data: {}'.format(in_pkl))
    with open(in_pkl, 'rb') as f:
        return pickle.load(f)


def load_data_by_split(args):
    """
    Load text-to-SQL dataset released by Finegan-Dollak et. al. 2018.

    The dataset adopts two different types of splits (split by question or by query type).
    """

    def fill_in_variables(s, s_vars, variables, target):
        var_list = {}
        for v_key, v_val in s_vars.items():
            if len(v_val) == 0:
                for var in variables:
                    if var['name'] == v_key:
                        v_val = var['example']
            s = s.replace(v_key, v_val)
            var_list[v_key] = v_val
        for var in variables:
            if not var['name'] in s_vars:
                v_loc = var['location']
                if target == 'program' and (v_loc == 'sql-only' or v_loc == 'both'):
                    v_key = var['name']
                    v_val = var['example']
                    s = s.replace(v_key, v_val)
                    var_list[v_key] = v_val
        return s, var_list

    dataset = dict()
    in_json = os.path.join(args.data_dir, '{}.json'.format(args.dataset_name))
    with open(in_json) as f:
        content = json.load(f)
        for example in content:
            programs = example['sql']
            query_split = example['query-split']
            variables = example['variables']
            for sentence in example['sentences']:
                question_split = sentence['question-split']
                nl = sentence['text']
                s_variables = sentence['variables']
                exp = Text2SQLExample(OTHERS, args.dataset_name, 0)
                if not args.normalize_variables:
                    nl, var_list = fill_in_variables(nl, s_variables, variables, target='text')
                    exp.variables = var_list
                exp.text = nl
                for i, program in enumerate(programs):
                    if not args.normalize_variables:
                        program, _ = fill_in_variables(program, s_variables, variables, target='program')
                    if program.endswith(';'):
                        program = program[:-1].rstrip()
                    exp.add_program(program)
                split = question_split if args.question_split else query_split
                if not split in dataset:
                    dataset[split] = []
                dataset[split].append(exp)
    return dataset


def load_data_wikisql(args):
    """
    Load the WikiSQL dataset released by Zhong et. al. 2018, assuming that the data format has been
    changed by the script `data_processor_wikisql.py`.
    """
    in_dir = args.data_dir
    splits = ['train', 'dev', 'test']
    schema_graphs = load_schema_graphs_wikisql(in_dir, splits=splits)

    dataset = dict()
    for split in splits:
        dataset[split] = load_data_split_wikisql(in_dir, split, schema_graphs)
    dataset['schema'] = schema_graphs
    return dataset


def load_data_split_wikisql(in_dir, split, schema_graphs):
    in_json = os.path.join(in_dir, '{}.json'.format(split))
    data_split = []
    with open(in_json) as f:
        content = json.load(f)
        for example in content:
            db_name = example['table_id']
            text = example['question']
            exp = Text2SQLExample(WIKISQL, db_name, db_id=schema_graphs.get_db_id(db_name))
            program = example['query']
            program_ast = example['sql']
            if program.endswith(';'):
                program = program[:-1].rstrip()
            exp.text = text
            exp.add_program_official(program, program_ast)
            data_split.append(exp)
    return data_split


def load_data_spider(args):
    """
    Load the Spider dataset released by Yu et. al. 2018.
    Or load the SCFG dataset, which is in Spider-y format.
    """
    in_dir = args.data_dir
    dataset = dict()
    schema_graphs = load_schema_graphs_spider(in_dir, args.schema_augmentation_factor,
                                              augment_with_wikisql=args.augment_with_wikisql,
                                              random_field_order=args.random_field_order,
                                              db_dir=args.db_dir)
    if 'scfg' in args.dataset_name or 'nlperturb' in args.dataset_name or 'perturbeddb' in args.data_dir: #forcing it to just be dev
        print(f"* force special dataset to be loaded as spider split")
        dataset['dev'] = load_data_split_spider(in_dir, args.dataset_name, schema_graphs,
                                                augment_with_wikisql=False)
    else:
        dataset['train'] = load_data_split_spider(in_dir, 'train', schema_graphs, args.schema_augmentation_factor,
                                                  get_data_augmentation_tag(args),
                                                  augment_with_wikisql=args.augment_with_wikisql)
        dataset['dev'] = load_data_split_spider(in_dir, 'dev', schema_graphs,
                                                augment_with_wikisql=args.augment_with_wikisql)
    dataset['schema'] = schema_graphs

    fine_tune_set = load_data_split_spider(in_dir, 'fine-tune', schema_graphs,
                                           augment_with_wikisql=args.augment_with_wikisql)
    if fine_tune_set:
        dataset['fine-tune'] = fine_tune_set
    return dataset


def load_data_split_spider(in_dir, split, schema_graphs, schema_augmentation_factor=1, aug_tag='',
                           augment_with_wikisql=False):
    if split == 'train':
        in_json = os.path.join(in_dir, '{}.{}json'.format(split, aug_tag))
    else:
        in_json = os.path.join(in_dir, '{}.json'.format(split)) #if dev and scfg, then it will just be the .json file
        print(f"IN_JSON: {in_json}")
    if not os.path.exists(in_json):
        print('Warning: file {} not found.'.format(in_json))
        return None
    data_split = []
    num_train_exps_by_db = collections.defaultdict(int)
    with open(in_json) as f:
        content = json.load(f)
        for example in content:
            db_name = example['db_id']
            if split == 'train':
                num_train_exps_by_db[db_name] += 1
            exp = Text2SQLExample(SPIDER, db_name, db_id=schema_graphs.get_db_id(db_name))
            text = example['question']
            program = example['query']
            if program.endswith(';'):
                program = program[:-1].rstrip()
            exp.text = text
            if 'question_toks' in example:
                text_tokens = example['question_toks']
                exp.text_tokens = [t.lower() for t in text_tokens]
                exp.text_ptr_values = text_tokens
            program_ast = example['sql'] if 'sql' in example else None
            program_tokens = example['query_toks'] if 'query_toks' in example else None
            if program_tokens and program_tokens[-1] == ';':
                program_tokens = program_tokens[:len(program_tokens)-1]
            exp.add_program_official(program, program_ast, program_tokens)
            if 'tables' in example:
                # gt_tables = example['tables']
                # gt_table_names = example['table_names']
                dummy_gt_tables = range(0, len(example['tables'])) #this doesnt break preproc
                dummy_gt_table_names = example['tables']
                exp.add_gt_tables(dummy_gt_tables, dummy_gt_table_names)
            data_split.append(exp)

    if split == 'train' and schema_augmentation_factor > 1:
        aug_data_split = collections.defaultdict(list)
        for exp in data_split:
            for i in range(schema_augmentation_factor - 1):
                new_db_name = '{}-{}'.format(exp.db_name, i)
                new_db_id = schema_graphs.get_db_id(new_db_name)
                new_exp = AugmentedText2SQLExample(exp, new_db_name, new_db_id)
                aug_data_split[exp.db_name].append(new_exp)
        for db_name in aug_data_split:
            # import random
            # random.seed(100)
            # if num_train_exps_by_db[db_name] > 125:
            #     num_samples, j = 0, schema_augmentation_factor
            #     while (num_samples <= 0):
            #         num_samples = 125 * j - num_train_exps_by_db[db_name]
            #         j += 1
            #     data_split += random.sample(aug_data_split[db_name], k=num_samples)
            # else:
            data_split += aug_data_split[db_name]

    print('{} {} examples loaded'.format(len(data_split), split))

    if split in ['train', 'dev'] and augment_with_wikisql:
        data_dir = os.path.dirname(in_dir)
        wikisql_dir = os.path.join(data_dir, 'wikisql1.1')
        wikisql_split = load_data_split_wikisql(wikisql_dir, split, schema_graphs)
        data_split += wikisql_split
        print('{} {} examples loaded (+wikisql)'.format(len(data_split), split))

    return data_split


def load_parsed_sqls(args, augment_with_wikisql=False):
    data_dir = args.data_dir
    dataset = args.dataset_name
    norm_tag = get_norm_tag(args)
    in_json = os.path.join(data_dir, '{}.{}parsed.json'.format(dataset, norm_tag))
    if not os.path.exists(in_json):
        print('Warning: parsed SQL files not found!')
        return dict()
    with open(in_json) as f:
        parsed_sqls = json.load(f)
        print('{} parsed SQL queries loaded'.format(len(parsed_sqls)))

    if augment_with_wikisql:
        parent_dir = os.path.dirname(data_dir)
        wikisql_dir = os.path.join(parent_dir, 'wikisql1.1')
        wikisql_parsed_json = os.path.join(wikisql_dir, 'wikisql.parsed.json')
        with open(wikisql_parsed_json) as f:
            wikisql_parsed_sqls = json.load(f)
            print('{} parsed wikisql SQL queries loaded'.format(len(wikisql_parsed_sqls)))
        parsed_sqls.update(wikisql_parsed_sqls)
        print('{} parsed SQL queries loaded (+wikisql)'.format(len(parsed_sqls)))

    return parsed_sqls


def save_parsed_sqls(args, parsed_sqls):
    data_dir = args.data_dir
    dataset = args.dataset_name
    norm_tag = get_norm_tag(args)
    out_json = os.path.join(data_dir, '{}.{}parsed.json'.format(dataset, norm_tag))
    # save a copy of the parsed file before directly modifying it
    #shutil.copyfile(out_json, os.path.join('/tmp', '{}.{}parsed.json'.format(dataset, norm_tag))) #TODO: think this really needs to work ...
    with open(out_json, 'w') as o_f:
        json.dump(parsed_sqls, o_f, indent=4)
        print('parsed SQL queries dumped to {}'.format(out_json))
    shutil.copyfile(out_json, os.path.join('/tmp', '{}.{}parsed.json'.format(dataset, norm_tag))) #copy it here?



def load_vocabs(args):
    """
    :return text_vocab: tokens appeared in the natural language query and schema
    :return program_vocab: tokens appeared in the program used for program generation
    :return world_vocab: tokens in the program that does not come from the input natural language query nor the schema
            (which likely needed to be inferred from world knowledge)
    """
    if args.model == 'seq2seq':
        return load_vocabs_seq2seq(args)
    elif args.dataset_name == 'scfg_data' or args.dataset_name == 'spider_nlperturb_data':
        return load_vocabs_seq2seq_ptr(args)
    elif args.model in ['seq2seq.pg', 'sqlova', 'ratsql', 'sqlova.pt']:
        return load_vocabs_seq2seq_ptr(args)
    elif args.model in ['vase']:
        return load_vocabs_vase(args)
    else:
        raise NotImplementedError


def load_vocabs_vase(args):
    text_vocab_path = get_vocab_path(args, 'nlperturb')
    text_vocab = load_vocab(text_vocab_path, args.text_vocab_min_freq, tu=utils.get_trans_utils(args))
    program_vocab = sql_reserved_tokens if args.pretrained_transformer else sql_reserved_tokens_revtok
    global value_vocab
    print('* text vocab size = {}'.format(text_vocab.size ))
    print('* program vocab size = {}'.format(program_vocab.size))
    print('* value vocab size = {}'.format(value_vocab.size))
    vocabs = {
        'text': text_vocab,
        'program': program_vocab,
        'value': value_vocab
    }
    return vocabs


def load_vocabs_seq2seq(args):
    if args.share_vocab:
        vocab_path = get_vocab_path(args, 'full')
        vocab = load_vocab(vocab_path, args.vocab_min_freq,  tu=utils.get_trans_utils(args))
        text_vocab, program_vocab = vocab, vocab
    else:
        text_vocab_path = get_vocab_path(args, 'nlperturb')
        text_vocab = load_vocab(text_vocab_path, args.text_vocab_min_freq, tu=utils.get_trans_utils(args))
        program_vocab_path = get_vocab_path(args, 'cm')
        program_vocab = load_vocab(program_vocab_path, args.program_vocab_min_freq)

    print('* text vocab size = {}'.format(text_vocab.size))
    print('* program vocab size = {}'.format(program_vocab.size))
    vocabs = {
        'text': text_vocab,
        'program': program_vocab
    }
    return vocabs


def load_vocabs_seq2seq_ptr(args):
    text_vocab_path = get_vocab_path(args, 'nl')
    text_vocab_path = "/export/home/sp-behavior-checking/data/spider/spider.sqlova.question-split.dn.bert.nl.vocab"
    text_vocab = load_vocab(text_vocab_path, args.text_vocab_min_freq, tu=utils.get_trans_utils(args))
    program_vocab = sql_reserved_tokens if args.pretrained_transformer else sql_reserved_tokens_revtok

    print('* text vocab size = {}'.format(text_vocab.size))
    print('* program vocab size = {}'.format(program_vocab.size))
    print()
    vocabs = {
        'text': text_vocab,
        'program': program_vocab
    }
    return vocabs


def load_vocab(vocab_path, min_freq, tag='', func_token_index=None, tu=None):
    """
    :param vocab_path: path to vocabulary file.
    :param min_freq: minimum frequency of known vocabulary (does not apply to meta data tokens).
    :param tag: a tag to mark the purpose of the vocabulary
    :param functional_tokens: funtional tokens prepended to the vocabulary.
    :param tu: pre-trained transformer utility object to use.
    :return: token to id mapping and the reverse mapping.
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = Vocabulary(tag=tag, func_token_index=func_token_index, tu=tu)
        for line in f.readlines():
            line = line.rstrip()
            v, freq = line.rsplit('\t', 1)
            freq = int(freq)
            in_vocab = is_functional_token(v) or freq < 0 or freq >= min_freq
            vocab.index_token(v, in_vocab, check_for_seen_vocab=True)
        print('vocab size = {}, loaded from {} with frequency threshold {}'.format(vocab.size, vocab_path, min_freq))
    return vocab
