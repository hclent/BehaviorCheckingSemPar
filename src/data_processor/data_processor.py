"""
Preprocessing text-to-SQL dataset released by Finegan-Dollak et. al. 2018.
"""

import collections
import copy
from functools import reduce
import numpy as np
import scipy.sparse as ssp
import pickle
import os

import moz_sp.sql_tokenizer as sql_tokenizer
from moz_sp import denormalize, parse
from src.data_processor.data_loader import load_parsed_sqls, save_parsed_sqls
from src.data_processor.data_loader import load_vocabs
from src.data_processor.schema_loader import load_schema_graphs
from src.data_processor.data_utils import Text2SQLExample, AugmentedText2SQLExample
from src.data_processor.path_utils import get_processed_data_path, get_vocab_path
from src.data_processor.sql.sql_reserved_tokens import sql_reserved_tokens, sql_reserved_tokens_revtok
from src.data_processor.vocab_utils import functional_token_index, functional_tokens, Vocabulary
import src.data_processor.tokenizers as tok
import src.data_processor.vectorizers as vec
from src.eval.eval_constant_extraction import SchemaLinkingEvaluator #eval_const_f1
from src.utils.utils import SEQ2SEQ, SEQ2SEQ_PG, SQLOVA, RATSQL, VASE
import src.utils.utils as utils


START_TOKEN = functional_token_index['start_token']
EOS_TOKEN = functional_token_index['eos_token']
NUM_TOKEN = functional_token_index['num_token']
STR_TOKEN = functional_token_index['str_token']
RESERVED_TOKEN = sql_tokenizer.RESERVED_TOKEN


spider_dev_dbs = {
    'employee_hire_evaluation',
    'battle_death',
    'student_transcripts_tracking',
    'poker_player',
    'wta_1',
    'world_1',
    'dog_kennels',
    'tvshow',
    'museum_visit',
    'voter_1',
    'singer',
    'pets_1',
    'concert_singer',
    'real_estate_properties',
    'orchestra',
    'course_teach',
    'cre_Doc_Template_Mgt',
    'network_1',
    'flight_2',
    'car_1'
}

spider_empty_dbs = {
    'music_2',
    'scholar',
    'sakila_1',
    'yelp',
    'geo',
    'academic',
    'formula_1',
    'restaurants',
    'imdb'
}


def preprocess(args, dataset, process_splits=('train', 'dev', 'test'), print_aggregated_stats=False, verbose=False):
    """
    Data pre-processing for baselines that does only shallow processing on the schema.
    """
    text_tokenize, program_tokenize, post_process, table_utils = tok.get_tokenizers(args)
    parsed_programs = load_parsed_sqls(args, augment_with_wikisql=args.augment_with_wikisql)
    num_parsed_programs = len(parsed_programs)

    vocabs = load_vocabs(args)

    schema_graphs = dataset['schema']
    schema_graphs.lexicalize_graphs(
        tokenize=text_tokenize, normalized=(args.model_id in [VASE, SQLOVA, RATSQL]))

    # 32 dbs, 119 table pairs contain ambiguities
    # num_ambs = 0
    # amb_dbs = set()
    # for db_name in schema_graphs.db_index:
    #     schema_graph = schema_graphs[db_name]
    #     for key in schema_graph.foreign_key_index:
    #         if len(schema_graph.foreign_key_index[key]) > 1:
    #             print(schema_graph.get_table(key[0]).name, schema_graph.get_table(key[1]).name)
    #             for i, (f1, f2) in enumerate(schema_graph.foreign_key_index[key]):
    #                 print('Key pair {}: {}, {}'.format(i, schema_graph.get_field(f1).name,
    #                 schema_graph.get_field(f2).name))
    #             amb_dbs.add(schema_graph.base_name)
    #             num_ambs += 1
    # print('{} foreign key ambiguities'.format(num_ambs))
    # print('Foreign key ambiguity detected in {} databases'.format(len(amb_dbs)))
    # import pdb
    # pdb.set_trace()

    ############################
    # data statistics
    num_oov = 0
    num_examples = 0
    num_denormalization_failed = 0
    num_schema_truncated = 0
    num_picklist_matched = []
    max_ptr_span_size = 0
    num_text_tokens, num_input_tokens, num_cm_tokens, num_cm_wf_tokens = [], [], [], []
    ############################

    # parallel data
    for split in process_splits:
        if not split in dataset:
            print(f"{split} split not in dataset...")
            continue
        stats = preprocess_split(dataset, split, args, parsed_programs,
                                 text_tokenize, program_tokenize, post_process, table_utils,
                                 schema_graphs, vocabs, verbose=verbose)
        ############################
        # update data statistics
        num_oov_split = stats[0]
        num_denormalization_failed_split = stats[1]
        num_schema_truncated_split = stats[2]
        num_picklist_matched_split = stats[3]
        max_ptr_span_size_split = stats[4]
        num_text_tokens_split, num_input_tokens_split, num_cm_tokens_split, num_cm_wf_tokens_split = stats[5:]
        num_oov += num_oov_split
        num_examples += len(dataset[split])
        num_denormalization_failed += num_denormalization_failed_split
        num_schema_truncated += num_schema_truncated_split
        num_picklist_matched += num_picklist_matched_split
        if max_ptr_span_size_split > max_ptr_span_size:
            max_ptr_span_size = max_ptr_span_size_split
        num_text_tokens += num_text_tokens_split
        num_input_tokens += num_input_tokens_split
        num_cm_tokens += num_cm_tokens_split
        num_cm_wf_tokens += num_cm_wf_tokens_split
        ############################

    # if len(parsed_programs) > num_parsed_programs:
    #     save_parsed_sqls(args, parsed_programs)

    #FORCE SAVE THE PARSES
    parsed_json = os.path.join(args.data_dir, '{}.parsed.json'.format(args.dataset_name))
    if not os.path.exists(parsed_json):
        print(f"* save the parsed sqls !!! ")
        save_parsed_sqls(args, parsed_programs)


    if print_aggregated_stats:
        print_data_statistics(num_oov, num_examples, num_denormalization_failed, num_schema_truncated,
                              max_ptr_span_size, num_text_tokens, num_input_tokens, num_cm_tokens, num_cm_wf_tokens)

    out_pkl = get_processed_data_path(args)
    with open(out_pkl, 'wb') as o_f:
        pickle.dump(dataset, o_f)
        print('Processed data dumped to {}'.format(out_pkl))


def preprocess_split(dataset, split, args, parsed_programs, text_tokenize, program_tokenize, post_process, table_utils,
                     schema_graphs, vocabs, print_split_stats=True, cache_examples=False, verbose=False):

    print(f"*****SPLIT: {split}")
    data_split = dataset[split]

    print('processing {} examples from {}...'.format(len(data_split), split))
    ############################
    # data statistics
    num_oov = 0
    num_denormalization_failed = 0
    num_schema_truncated = 0
    num_picklist_matched = []
    max_ptr_span_size = 0
    micro_v_prec, micro_v_recall, micro_v_f1 = [], [], []
    num_text_tokens, num_input_tokens, num_cm_tokens, num_cm_wf_tokens = [], [], [], []
    ############################

    for i, example in enumerate(data_split):
        print(f"type example in preprocess_split: {type(example)}")
        schema_graph = schema_graphs.get_schema(example.db_id)
        # if schema_graph.name != 'voter_1':
        #     continue
        query_oov, denormalized, schema_truncated, matched_values = preprocess_example(split, example, args,
                                                                                       parsed_programs,
                                                                                       text_tokenize,
                                                                                       program_tokenize,
                                                                                       post_process,
                                                                                       table_utils,
                                                                                       schema_graph,
                                                                                       vocabs,
                                                                                       verbose=verbose)
        # evaluate value extraction
        ground_truth_values = extract_value_spans(example.program_singleton_field_tokens,
                                                  example.program_singleton_field_token_types,
                                                  table_utils)
        pred_values = matched_values.values()
        prec, recall, f1 = SchemaLinkingEvaluator.eval_const_f1(ground_truth_values, pred_values)
        micro_v_prec.append(prec)
        micro_v_recall.append(recall)
        micro_v_f1.append(f1)

        # update data statistics
        ############################
        if query_oov:
            num_oov += 1
        if not denormalized:
            num_denormalization_failed += 1
        if schema_truncated:
            num_schema_truncated += 1
        if matched_values:
            num_picklist_matched.append(len(matched_values))
        ############################

        if split == 'train' and isinstance(example, Text2SQLExample):
            for var_val in example.variables.values():
                var_tokens = text_tokenize(var_val)
                if len(var_tokens) > max_ptr_span_size:
                    max_ptr_span_size = len(var_tokens)
        num_text_tokens.append(example.num_text_tokens)
        num_input_tokens.append(example.num_input_tokens)
        num_cm_tokens.append(example.num_program_tokens)
        num_cm_wf_tokens.append(len(example.program_singleton_field_input_ids))

        if i > 0 and i % 100000 == 0:
            print('{} examples processed'.format(i))
            if cache_examples:
                with open('temp_{}.pkl'.format(i), 'wb') as o_f:
                    pickle.dump(data_split, o_f)

    print('--- Value extraction performance ---')
    print('micro precision = {}'.format(np.mean(micro_v_prec)))
    print('micro recall = {}'.format(np.mean(micro_v_recall)))
    print('micro F1 = {}'.format(np.mean(micro_v_f1)))
    if print_split_stats:
        print('********** {} Data Statistics ***********'.format(split))
        print_data_statistics(num_oov, len(data_split),
                              num_denormalization_failed,
                              num_schema_truncated,
                              num_picklist_matched,
                              max_ptr_span_size,
                              num_text_tokens,
                              num_input_tokens,
                              num_cm_tokens,
                              num_cm_wf_tokens)

    return num_oov, num_denormalization_failed, num_schema_truncated, num_picklist_matched, max_ptr_span_size, \
           num_text_tokens, num_input_tokens, num_cm_tokens, num_cm_wf_tokens


def extract_value_spans(program_tokens, program_token_types, tu):
    values = []
    value, is_value = [], False
    for t, t_type in zip(program_tokens, program_token_types):
        if t_type == sql_tokenizer.VALUE:
            value.append(t)
        else:
            if value:
                value_str = tu.tokenizer.convert_tokens_to_string(value)
                if not utils.is_number(value_str):
                    values.append(value_str)
                value = []
    return values


def get_table_aware_transformer_encoder_inputs(text_tokens, text_features, schema_features, tu):
    num_excluded_tables, num_excluded_fields = 0, 0
    num_separators = 3
    max_schema_features_len = tu.tokenizer.max_len - num_separators - len(text_tokens)
    if len(schema_features) > max_schema_features_len:
        truncate_id = -1
        for i in range(len(schema_features)-1, -1, -1):
            if schema_features[i] == tu.table_marker:
                num_excluded_tables += 1
            elif schema_features[i] in [tu.field_marker, tu.primary_key_marker]:
                num_excluded_fields += 1
            else:
                if schema_features[i] != tu.value_marker and i < max_schema_features_len:
                    truncate_id = i
                    break
        if truncate_id > 0:
            schema_features = schema_features[:(truncate_id + 1)]
    input_tokens = [tu.cls_token] + text_features + [tu.sep_token] + schema_features + [tu.sep_token]
    input_ptr_values = [tu.cls_token] + text_tokens + [tu.sep_token] + schema_features + [tu.sep_token]
    return input_tokens, input_ptr_values, num_excluded_tables, num_excluded_fields


def get_transformer_output_value_mask(features, matched_values, tu):
    mask = []
    value_strs = list(matched_values.values())
    value, value_features, value_tokens = [], [], []
    is_value = False
    for i, x in enumerate(features):
        if is_value:
            if x in [tu.table_marker, tu.field_marker, tu.value_marker, tu.sep_token, tu.cls_token, tu.pad_token]:
                if value:
                    value_features.append(value)
                    value_tokens.append(utils.restore_feature_case(value, value_strs[len(value_tokens)]))
                    value = []
                mask.append(0)
                if x != tu.value_marker:
                    is_value = False
            else:
                mask.append(1)
                value.append(x)
        else:
            if x == tu.value_marker:
                is_value = True
            mask.append(0)
    # Notes: no need to check for additional values here as the input is guaranteed to end with '[SEP]'
    assert(len(value) == 0)
    if len(value_tokens) != len(matched_values):
        print('Warning: not all matched values included in schema encoding')
    assert(len(value_tokens) == len(value_features))
    if value_features:
        value_features = reduce(lambda x, y: x + y, value_features)
        value_tokens = reduce(lambda x, y: x + y, value_tokens)
    return mask, value_features, value_tokens


def preprocess_example(split, example, args, parsed_programs, text_tokenize, program_tokenize, post_process,
                       table_utils, schema_graph, vocabs, verbose=False):
    tu = table_utils
    text_vocab = vocabs['text']
    program_vocab = vocabs['program']
    if args.model_id in [VASE]:
        value_vocab = vocabs['value']

    def get_memory_values(features, raw_text, args):
        if args.pretrained_transformer.startswith('bert-') and args.pretrained_transformer.endswith('-uncased'):
            return utils.restore_feature_case(features, raw_text)
        else:
            return features

    def get_text_schema_adjacency_matrix(text_features, s_M):
        schema_size = s_M.shape[0]
        text_size = len(text_features)
        full_size = schema_size + text_size
        M = ssp.lil_matrix((full_size, full_size), dtype=np.int)
        M[-schema_size:, -schema_size:] = s_M
        return M

    # sanity check
    ############################
    query_oov = False
    denormalized = False
    schema_truncated = False
    matched_values = None
    ############################

    # Text feature extraction and set program ground truth list
    if isinstance(example, Text2SQLExample):
        if args.pretrained_transformer:
            text_features = text_tokenize(example.text)
            text_tokens = get_memory_values(text_features, example.text, args)
        else:
            text_tokens = text_tokenize(example.text, functional_tokens)
            text_features = [t.lower() for t in text_tokens]
        example.text_tokens = text_features
        example.text_ptr_values = text_tokens
        example.text_ids = vec.vectorize(text_features, text_vocab)
        example.text_ptr_input_ids = vec.vectorize(text_features, text_vocab)
        program_list = example.program_list
    else:
        text_tokens = example.example.text_ptr_values
        text_features = example.example.text_tokens
        program_list = example.example.program_list

    # Schema feature extraction
    if args.model_id in [VASE, SQLOVA, RATSQL]:
        question_encoding = example.text if args.use_picklist else None
        gt_tables = sorted([schema_graph.get_table_id(t_name) for t_name in example.gt_table_names]) \
            if args.use_oracle_tables else None
        schema_features, matched_values = schema_graph.get_serialization(
            tu, flatten_features=True, tables=gt_tables, use_typed_field_markers=args.use_typed_field_markers,
            use_graph_encoding=args.use_graph_encoding, question_encoding=question_encoding,
            top_k_matches=args.top_k_picklist_matches, num_values_per_field=args.num_values_per_field,
            no_anchor_text=args.no_anchor_text)
        example.input_tokens, example.input_ptr_values, num_excluded_tables, num_excluded_fields = \
            get_table_aware_transformer_encoder_inputs(text_tokens, text_features, schema_features, table_utils)
        schema_truncated = (num_excluded_fields > 0)
        num_included_nodes = schema_graph.get_num_nodes(gt_tables) + 1 - num_excluded_tables - num_excluded_fields
        example.ptr_input_ids = vec.vectorize(example.input_tokens, text_vocab)
        if args.read_picklist:
            example.transformer_output_value_mask, value_features, value_tokens = \
                get_transformer_output_value_mask(example.input_tokens, matched_values, tu)
        example.primary_key_ids = schema_graph.get_primary_key_ids(num_included_nodes, tables=gt_tables)
        example.foreign_key_ids = schema_graph.get_foreign_key_ids(num_included_nodes, tables=gt_tables)
        example.field_type_ids = schema_graph.get_field_type_ids(num_included_nodes, tables=gt_tables)
        example.table_masks = schema_graph.get_table_masks(num_included_nodes, tables=gt_tables)
        example.field_table_pos = schema_graph.get_field_table_pos(num_included_nodes, tables=gt_tables)
        example.schema_M = schema_graph.adj_matrix
        example.M = get_text_schema_adjacency_matrix(text_features, example.schema_M)
    else:
        num_included_nodes = schema_graph.num_nodes

    # Value copy feature extraction
    if args.read_picklist:
        constant_memory_features = text_features + value_features
        constant_memory = text_tokens + value_tokens
        example.text_ptr_values = constant_memory
    else:
        constant_memory_features = text_features
    constant_ptr_value_ids, constant_unique_input_ids = vec.vectorize_ptr_in(
        constant_memory_features, program_vocab, in_out_no_overlap=(args.model_id in [VASE, SQLOVA, RATSQL]))
    if isinstance(example, Text2SQLExample):
        example.text_ptr_value_ids = constant_ptr_value_ids
    example.ptr_value_ids = constant_ptr_value_ids + [program_vocab.size + len(constant_memory_features) + x
                                                      for x in range(num_included_nodes)]

    if not args.leaderboard_submission:
        for j, program in enumerate(program_list):
            if isinstance(example, Text2SQLExample):
                ast, denormalized = get_ast(program, parsed_programs, args.denormalize_sql, schema_graph)
                if not ast:
                    ast = program
                example.program_ast_list.append(ast)
                # if example.num_programs != len(example.program_ast_list):
                #     import pdb
                #     pdb.set_trace()
                program_tokens = program_tokenize(ast,
                                                  schema=schema_graph,
                                                  omit_from_clause=args.omit_from_clause,
                                                  no_join_condition=args.no_join_condition,
                                                  in_execution_order=args.process_sql_in_execution_order)
                assert(len(program_tokens) > 0)
                program_tokens = [START_TOKEN] + program_tokens + [EOS_TOKEN]
                program_input_ids = vec.vectorize(program_tokens, program_vocab)
                example.program_input_ids_list.append(program_input_ids)

                # Model I. Vanilla pointer-generator output
                if args.model_id in [SEQ2SEQ_PG]:
                    program_text_ptr_value_ids = vec.vectorize_ptr_out(program_tokens, program_vocab,
                                                                       constant_unique_input_ids)
                    example.program_text_ptr_value_ids_list.append(program_text_ptr_value_ids)
                    # sanity check
                    #   NL pointer output contains tokens that does not belong to any of the following categories
                    #     - reserved tokens
                    #     - tokens in the NL input
                    #     - tokens from environment variables (e.g. table schema)
                    ############################
                    if program_vocab.unk_id in program_text_ptr_value_ids:
                        # unk_indices = [i for i, x in enumerate(program_text_ptr_value_ids) if x == program_vocab.unk_id]
                        # print('OOV I: {}'.format(' '.join([program_tokens[i] for i in unk_indices])))
                        # example.pretty_print(schema=schema_graph,
                        #                      de_vectorize_ptr=vec.de_vectorize_ptr,
                        #                      de_vectorize_field_ptr=vec.de_vectorize_field_ptr,
                        #                      rev_vocab=program_vocab,
                        #                      post_process=post_process)
                        query_oov = True
                    ############################

                # Model II. SQLova output
                assert(ast is not None)
                denormalized_ast, _ = denormalize(ast, schema_graph, return_parse_tree=True)
                example.program_denormalized_ast_list.append(denormalized_ast)
                tokenizer_output = program_tokenize(denormalized_ast,
                                                    return_token_types=True,
                                                    schema=schema_graph,
                                                    keep_singleton_fields=True,
                                                    omit_from_clause=args.omit_from_clause,
                                                    no_join_condition=args.no_join_condition,
                                                    atomic_value=(args.model_id in [VASE]),
                                                    num_token=NUM_TOKEN, str_token=STR_TOKEN,
                                                    in_execution_order=args.process_sql_in_execution_order)
                program_singleton_field_tokens = tokenizer_output[0]
                program_singleton_field_token_types = tokenizer_output[1]
                program_singleton_field_tokens = [START_TOKEN] + program_singleton_field_tokens + [EOS_TOKEN]
                program_singleton_field_token_types = \
                    [RESERVED_TOKEN] + program_singleton_field_token_types + [RESERVED_TOKEN]
                example.program_singleton_field_tokens_list.append(program_singleton_field_tokens)
                example.program_singleton_field_token_types_list.append(program_singleton_field_token_types)
                program_singleton_field_input_ids = vec.vectorize_singleton(
                    program_singleton_field_tokens, program_singleton_field_token_types, program_vocab)
                example.program_singleton_field_input_ids_list.append(program_singleton_field_input_ids)

                # Model III. VASE output
                if args.model_id in [VASE]:
                    constants = tokenizer_output[2]
                    example.leaf_condition_vals_list.append(constants)
                    leaf_condition_val_ids = []
                    for constant_tokens, _ in constants:
                        constant_tokens_ = [START_TOKEN] + constant_tokens + [EOS_TOKEN]
                        leaf_condition_val_ids.append(vec.vectorize(constant_tokens_, value_vocab))
                    example.leaf_condition_val_ids_list.append(leaf_condition_val_ids)
            else:
                # Model II. SQLova output
                example.program_singleton_field_input_ids_list.append(
                    example.example.program_singleton_field_input_ids_list[j])
                program_singleton_field_tokens = example.example.program_singleton_field_tokens_list[j]
                program_singleton_field_token_types = example.example.program_singleton_field_token_types_list[j]

                # Model III. VASE output
                if args.model_id in [VASE]:
                    example.leaf_condition_val_ids_list.append(example.example.leaf_condition_val_ids_list[j])
                    constants = example.example.leaf_condition_vals_list[j]

            program_field_ptr_value_ids = vec.vectorize_field_ptr_out(program_singleton_field_tokens,
                                                                      program_singleton_field_token_types,
                                                                      program_vocab,
                                                                      constant_unique_input_ids,
                                                                      max_memory_size=len(constant_memory_features),
                                                                      schema=schema_graph,
                                                                      num_included_nodes=num_included_nodes)
            example.program_text_and_field_ptr_value_ids_list.append(program_field_ptr_value_ids)

            # Model III. VASE output
            if args.model_id in [VASE]:
                if isinstance(example, Text2SQLExample):
                    constant_ptr_value_ids, constant_unique_input_ids = vec.vectorize_ptr_in(
                        constant_memory_features, value_vocab, in_out_no_overlap=True)
                    example.text_ptr_value_ids = constant_ptr_value_ids
                else:
                    _, constant_unique_input_ids = vec.vectorize_ptr_in(text_features, value_vocab, in_out_no_overlap=True)
                leaf_condition_ptr_val_ids = []
                for constant_tokens, constant_token_types in constants:
                    constant_tokens_ = [START_TOKEN] + constant_tokens + [EOS_TOKEN]
                    constant_token_types_ = [RESERVED_TOKEN] + constant_token_types + [RESERVED_TOKEN]
                    leaf_condition_ptr_val_ids.append(
                        vec.vectorize_field_ptr_out(constant_tokens_, constant_token_types_, value_vocab,
                                                    constant_unique_input_ids, max_memory_size=len(constant_memory_features)))
                example.leaf_condition_val_ids_list.append(leaf_condition_ptr_val_ids)




            #table_ids = [schema_graph.get_table_id(table_name) for table_name in example.gt_table_names_list[j]]
            table_ids = [num for num in range(0, len(example.gt_table_names))]
            example.table_ids_list.append(table_ids)
            #assert ([schema_graph.get_table(x).name for x in table_ids] == example.gt_table_names)

            # sanity check
            ############################
            #   NL+Schema pointer output contains tokens that does not belong to any of the following categories
            if verbose:
                if program_vocab.unk_id in program_field_ptr_value_ids:
                    unk_indices = [i for i, x in enumerate(program_field_ptr_value_ids) if x == program_vocab.unk_id]
                    print('OOV II: {}'.format(' '.join([program_singleton_field_tokens[i] for i in unk_indices])))
                    example.pretty_print(schema=schema_graph,
                                         de_vectorize_ptr=vec.de_vectorize_ptr,
                                         de_vectorize_field_ptr=vec.de_vectorize_field_ptr,
                                         rev_vocab=program_vocab,
                                         post_process=post_process,
                                         use_table_aware_te=(args.model_id in [VASE, SQLOVA, RATSQL]))
                    query_oov = True
            if program_vocab.unk_field_id in program_field_ptr_value_ids:
                example.pretty_print(schema=schema_graph,
                                     de_vectorize_ptr=vec.de_vectorize_ptr,
                                     de_vectorize_field_ptr=vec.de_vectorize_field_ptr,
                                     rev_vocab=program_vocab,
                                     post_process=post_process,
                                     use_table_aware_te=(args.model_id in [VASE, SQLOVA, RATSQL]))
                # import pdb
                # pdb.set_trace()
            if program_vocab.unk_table_id in program_field_ptr_value_ids:
                example.pretty_print(schema=schema_graph,
                                     de_vectorize_ptr=vec.de_vectorize_ptr,
                                     de_vectorize_field_ptr=vec.de_vectorize_field_ptr,
                                     rev_vocab=program_vocab,
                                     post_process=post_process,
                                     use_table_aware_te=(args.model_id in [VASE, SQLOVA, RATSQL]))
                # import pdb
                # pdb.set_trace()
            ############################

            # Store the ground truth queries after preprocessing to run a relaxed evaluation or
            # to evaluate with partial queries
            # example.gt_program_list.append(program)
            if split == 'dev':
                input_tokens = text_tokens
                if args.model_id in [VASE, SQLOVA, RATSQL]:
                    _p = vec.de_vectorize_field_ptr(program_field_ptr_value_ids, program_vocab, input_tokens,
                                                    schema=schema_graph, post_process=post_process)
                elif args.model_id in [SEQ2SEQ_PG]:
                    _p = vec.de_vectorize_ptr(program_text_ptr_value_ids, program_vocab, input_tokens,
                                              post_process=post_process)
                else:
                    _p = program
                example.gt_program_list.append(_p)

            # sanity check
            ############################
            # try:
            #     assert(equal_ignoring_trivial_diffs(_p, program.lower(), verbose=True))
            # except Exception:
            #     print('_p:\t\t{}'.format(_p))
            #     print('program:\t{}'.format(program))
            #     print()
            #     import pdb
            #     pdb.set_trace()
            ############################

        example.run_unit_tests()

    return query_oov, denormalized, schema_truncated, matched_values


def get_ast(program, parsed_programs=None, denormalize_sql=False, schema_graph=None):
    ast = parsed_programs.get(program, None) if parsed_programs else None
    if ast is None:
        try:
            ast = parse(program)
            print('SQL query parsed: {}'.format(program))
            parsed_programs[program] = ast
        except Exception:
            print('SQL query cannot be parsed: {}'.format(program))
    denormalized = False
    if denormalize_sql:
        if ast:
            ast, _ = denormalize(copy.deepcopy(ast), schema_graph, return_parse_tree=True)
            denormalized = True
    return ast, denormalized


def build_vocab(args, dataset, schema_graphs):
    """
    Construct vocabularies.

    This function saves to disk:
    - text vocab: consists of tokens appeared in the natural language query and schema
    - program vocab: consists of tokens appeared in the program
    - schema vocab: consists of table and field names from the schema
    - world vocab: consists of tokens in the program that does not come from any of the above category
      (which likely needed to be inferred from world knowledge)
    """
    print('Constructing vocabulary...')

    text_tokenize, program_tokenize, _, tu = tok.get_tokenizers(args)
    if args.pretrained_transformer:
        sql_reserved_vocab = sql_reserved_tokens
    else:
        sql_reserved_vocab = sql_reserved_tokens_revtok
    parsed_programs = load_parsed_sqls(args, augment_with_wikisql=args.augment_with_wikisql)

    schema_graphs.lexicalize_graphs(
        tokenize=text_tokenize, normalized=(args.model_id in [VASE, SQLOVA, RATSQL]))

    # compute text and program vocab
    text_hist, program_hist = collections.defaultdict(int), collections.defaultdict(int)
    world_vocab = Vocabulary('world')

    for split in ['train', 'dev', 'test']:
        if not split in dataset:
            continue
        data_split = dataset[split]
        for i, example in enumerate(data_split):
            if isinstance(example, AugmentedText2SQLExample):
                continue
            schema_graph = schema_graphs.get_schema(example.db_id)
            text = example.text
            if args.pretrained_transformer:
                text_tokens = text_tokenize(text)
            else:
                text_tokens = text_tokenize(text.lower(), functional_tokens)
            for word in text_tokens:
                text_hist[word] += 1
            for program in example.program_list:
                ast, _ = get_ast(program, parsed_programs, args.denormalize_sql, schema_graph)
                if ast:
                    program = ast
                program_tokens = program_tokenize(program, omit_from_clause=args.omit_from_clause,
                                                  no_join_condition=args.no_join_condition)
                for token in program_tokens:
                    program_hist[token] += 1
                    if split == 'train':
                        if not token in text_tokens and not sql_reserved_vocab.contains(token):
                            world_vocab.index_token(token, in_vocab=True)
            if i > 0 and i % 5000 == 0:
                print('{} examples processed'.format(i))

    if args.pretrained_transformer.startswith('bert') or args.pretrained_transformer == 'table-bert':
        text_hist = dict()
        for v in tu.tokenizer.vocab:
            text_hist[v] = tu.tokenizer.vocab[v]
        for v in tu.tokenizer.added_tokens_encoder:
            text_hist[v] = tu.tokenizer.convert_tokens_to_ids(v)
        schema_lexical_vocab = None
    elif args.pretrained_transformer.startswith('roberta'):
        text_hist = tu.tokenizer.encoder
        schema_lexical_vocab = None
    else:
        schema_lexical_vocab = schema_graphs.get_lexical_vocab()

    export_vocab(text_hist, program_hist, schema_lexical_vocab, world_vocab, args)


def export_vocab(text_hist, program_hist, schema_lexical_vocab, world_vocab, args):

    if schema_lexical_vocab is not None:
        # Merge the lexicon based on the natural language text and the database schema
        for v in schema_lexical_vocab:
            text_hist[v] = -1

    text_vocab = Vocabulary('text', func_token_index=functional_token_index, tu=utils.get_trans_utils(args))
    full_vocab = Vocabulary('full', func_token_index=functional_token_index, tu=utils.get_trans_utils(args))
    for v in text_hist:
        text_vocab.index_token(v, True, text_hist[v])
    text_vocab_path = get_vocab_path(args, 'nlperturb')
    text_vocab.save_to_disk(text_vocab_path)

    program_vocab = Vocabulary('program', func_token_index=functional_token_index)
    for v in program_hist:
        program_vocab.index_token(v, True, program_hist[v])
    program_vocab_path = get_vocab_path(args, 'cm')
    program_vocab.save_to_disk(program_vocab_path)

    # Combine text and program vocabularies
    full_vocab.merge_with(text_vocab)
    full_vocab.merge_with(program_vocab)
    full_vocab_path = get_vocab_path(args, 'full')
    full_vocab.save_to_disk(full_vocab_path)

    world_vocab_path = get_vocab_path(args, 'world')
    world_vocab.save_to_disk(world_vocab_path)


def demo_preprocess(args, example, vocabs=None, schema_graph=None):
    text_tokenize, program_tokenize, post_process, tu = tok.get_tokenizers(args)
    if not schema_graph:
        schema_graphs = load_schema_graphs(args)
        schema_graph = schema_graphs.get_schema(example.db_id)
    schema_graph.lexicalize_graph(tokenize=text_tokenize,
                                  normalized=(args.model_id in [VASE, SQLOVA, RATSQL]))

    question_encoding = example.text if args.use_picklist else None
    gt_tables = sorted([schema_graph.get_table_id(t_name) for t_name in example.gt_table_names]) \
        if args.use_oracle_tables else None
    schema_features = schema_graph.get_serialization(
        tu, flatten_features=True, tables=gt_tables, use_typed_field_markers=args.use_typed_field_markers,
        use_graph_encoding=args.use_graph_encoding, question_encoding=question_encoding,
        top_k_matches=args.top_k_picklist_matches, num_values_per_field=args.num_values_per_field,
        no_anchor_text=args.no_anchor_text)

    preprocess_example('test', example, args, {}, text_tokenize, program_tokenize, post_process, tu,
                       schema_graph, vocabs)


def print_data_statistics(num_oov, num_examples, num_denormalization_failed, num_schema_truncated, num_picklist_matched,
                          max_ptr_span_size, num_text_tokens, num_input_tokens, num_cm_tokens, num_cm_wf_tokens):
    print('OOV observed in {}/{} examples'.format(num_oov, num_examples))
    print('Denormalization skipped for {} examples'.format(num_denormalization_failed))
    print('Schema truncated for {} examples'.format(num_schema_truncated))
    print('Picklist matched for {} examples ({} matches per example)'.format(
        len(num_picklist_matched), np.mean(num_picklist_matched)))
    if len(num_text_tokens) > 0:
        print('+ text sizes')
        print('# text tokens (avg) = {}'.format(np.mean(num_text_tokens)))
        print('# text tokens (min) = {}'.format(np.min(num_text_tokens)))
        print('# text tokens (max) = {}'.format(np.max(num_text_tokens)))
        print('+ input sizes')
        print('input size (avg) = {}'.format(np.mean(num_input_tokens)))
        print('input size (min) = {} '.format(np.min(num_input_tokens)))
        print('input size (max) = {}'.format(np.max(num_input_tokens)))
        print('+ program sizes')
        print('# program tokens (avg) = {}\t# program whole field tokens = {} (avg)\t'.format(
            np.mean(num_cm_tokens), np.mean(num_cm_wf_tokens)))
        print('# program tokens (min) = {}\t# program whole field tokens = {} (min)\t'.format(
            np.min(num_cm_tokens), np.min(num_cm_wf_tokens)))
        print('# program tokens (max) = {}\t# program whole field tokens = {} (max)\t'.format(
            np.max(num_cm_tokens), np.max(num_cm_wf_tokens)))
        print('max pointer span size = {}'.format(max_ptr_span_size))
