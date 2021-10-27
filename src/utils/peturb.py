import os, sys
import json
import re
import random
import copy
import spacy
from itertools import chain
import inflect
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.wsd import lesk
from random import randint
import nltk.data
from sqlalchemy import create_engine, Table, MetaData


from src.data_processor.data_utils import WIKISQL
from src.data_processor.data_utils import SPIDER
from src.eval.process_sql import tokenize, get_schema, get_tables_with_alias, get_table_recur, Schema, get_sql
from moz_sp import tokenize as mozsptokenize
import src.utils.trans.bert_utils as bu
from collections import defaultdict

random.seed(27)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') #for db perturbation... #TODO: revisit if this is needed?
lemmatizer = WordNetLemmatizer()
p = inflect.engine()


sql_stopwords = ['and', 'as', 'asc', 'between', 'case', 'collate_nocase', 'cross_join', 'desc', 'else', 'end', 'from',
                 'full_join', 'full_outer_join', 'group_by', 'having', 'in', 'inner_join', 'is', 'is_not', 'join',
                 'left_join', 'left_outer_join', 'like', 'limit', 'none', 'not_between', 'not_in', 'not_like', 'offset',
                 'on', 'or', 'order_by', 'reserved', 'right_join', 'right_outer_join', 'select', 'then', 'union',
                 'union_all', 'except', 'intersect', 'using', 'when', 'where', 'binary_ops', 'unary_ops', 'with',
                 'durations', 'max', 'min', 'count', 'sum', 'avg', 'minimum', 'maximum', 'ascending', 'descending', 'average']

def getTrueSchemaAsDict(db):
    true_schema_graph = defaultdict(list)

    old_db_dir = "/export/home/sp-behavior-checking/data/spider/database"
    old_full_path = os.path.join(old_db_dir, db, db + ".sqlite")
    # read in the old. copy things like meta data
    old_engine = create_engine(f"sqlite:///{old_full_path}")
    old_metadata = MetaData(bind=old_engine)
    old_metadata.reflect(old_engine)
    for t, something in old_metadata.tables.items():
        tObj = Table(f"{t}", old_metadata)
        for col in tObj.columns:
            true_schema_graph[t].append(col.name)
    return true_schema_graph

def get_random_db_schema(db):
    #NOTE: it makes sense to stick to dev_dbs because I know that they do not have issues with tables/cols having spaces
    # or unicode problems
    dev_dbs = ['dog_kennels', 'concert_singer', 'cre_Doc_Template_Mgt', 'employee_hire_evaluation', 'orchestra',
               'network_1', 'pets_1', 'flight_2', 'poker_player', 'voter_1', 'student_transcripts_tracking',
               'museum_visit', 'world_1', 'tvshow', 'battle_death', 'real_estate_properties', 'course_teach',
               'singer', 'wta_1', 'car_1']
    candidate_dbs = [i for i in dev_dbs if i is not db]
    rand_db = random.choice(candidate_dbs)
    random_db_schema = getTrueSchemaAsDict(rand_db)
    return rand_db, random_db_schema

def get_inUse_tables_and_columns(gt_schema, tokens):
        in_use_tables = [t for t in gt_schema.keys() if t in tokens]
        in_use_columns = {} #need to know the table associated with the inuse columns, in case multiple tables are in use!
        for t in in_use_tables:
            cols = gt_schema[t]
            cols_in_tokens = [c for c in cols if c in tokens]
            if len(cols_in_tokens) > 0:
                in_use_columns[t] = cols_in_tokens
            else: #if none of them are in the tokens just add them all
                in_use_columns[t] = cols
                #[in_use_columns.append(c) for c in cols]
        return in_use_tables, in_use_columns

def is_camel_case(s): #Camel case AND Pascal case will pass
  if s != s.lower() and s != s.upper() and "_" not in s and sum(i.isupper() for i in s) > 1:
      return True
  elif s == "ratingDate": #ugly hardcoding, but sorry
      return True
  else:
    return False

def is_snake_case(s):
  #if s == s.lower() and s != s.upper() and "_" in s:
  if "_" in s:
      return True
  return False

def changeToKebab(item):
    if is_camel_case(item):
        tricky = ["StuID", "PetID", "GNPOld"]
        trickyLUT = {"StuID": "Stu-ID", "PetID": "Pet-ID", "GNPOld": "GNP-Old"}
        if item not in tricky:
            peturbed_table_name = re.sub(r'(?<!^)(?=[A-Z])', '-', item)
        else:
            peturbed_table_name = trickyLUT[item]
    elif is_snake_case(item):
        peturbed_table_name = re.sub('_', '-', item)
    else:
        peturbed_table_name = item #cant do anything with it
    return peturbed_table_name

def changeToCamel(item):
    if is_camel_case(item):
        return item
    elif is_snake_case(item): #this is so hacky, I appologize world
        buffer = []
        buffer.append(item[0])
        for i in range(1, len(item)):
            if item[i-1] == "_":
                buffer.append(item[i].upper())
            else:
                if item[i] != "_":
                    buffer.append(item[i])
        new_name = ('').join(buffer)
        return new_name
    else:
        return item

def changeToSpaced(item):
    if is_camel_case(item):
        tricky = ["StuID", "PetID", "GNPOld"]
        trickyLUT = {"StuID": "Stu ID", "PetID": "Pet ID", "GNPOld": "GNP Old"}
        if item not in tricky:
            peturbed_table_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', item)
        else:
            peturbed_table_name = trickyLUT[item]
    elif is_snake_case(item):
        peturbed_table_name = re.sub('_', ' ', item)
    else:
        peturbed_table_name = item #cant do anything with it
    return peturbed_table_name

def perturb_change_case(gt_schema, case="kebab"):
    """
    gt_schema: Dict{table: [list of columns]}
    case: String. What kind of casing should the new DB have? Options: kebab, camel, space
    """
    new_schema = {}
    perturbed_key = {} #'original': 'perTurbed'
    reverse_perturbed_key = {} #'perTurbed': original

    assert case in ["kebab", "camel", "space"]

    for table, column_list in gt_schema.items():
        perturbed_column_list = []

        table_to_perturb = table

        if case == "kebab":
            peturbed_table_name = changeToKebab(table_to_perturb)
        elif case == "camel":
            peturbed_table_name = changeToCamel(table_to_perturb)
        elif case == "space":
            peturbed_table_name = changeToSpaced(table_to_perturb)
        else:
            peturbed_table_name = table_to_perturb

        perturbed_key[table] = peturbed_table_name
        reverse_perturbed_key[peturbed_table_name] = table

        for col in column_list:
            col_to_perturb = col
            if case == "kebab":
                peturbed_col_name = changeToKebab(col_to_perturb)
            elif case == "camel":
                peturbed_col_name = changeToCamel(col_to_perturb)
            elif case == "space":
                peturbed_col_name = changeToSpaced(col_to_perturb)
            else:
                peturbed_col_name = col_to_perturb

            perturbed_key[col] = peturbed_col_name
            reverse_perturbed_key[peturbed_col_name] = col
            perturbed_column_list.append(peturbed_col_name)

        assert len(perturbed_column_list) == len(gt_schema[table])
        new_schema[peturbed_table_name] = perturbed_column_list
    assert len(new_schema.keys()) == len(gt_schema.keys())

    return new_schema, perturbed_key, reverse_perturbed_key

def naiveTokenize(name):
    """
    input: table or column name
    return: table/column name split up into "words
    """
    spaced_case = changeToSpaced(name)
    tokens = spaced_case.split(" ")
    if len(tokens) == 1:
        return None
    else:
        return tokens

def checkForAbbrev(tokens, original):
    abbrevDic = {'id': 'identification', 'ids': 'identifications',
                 'tv': 'television', 'wta': "Womens Tenis Association",
                 'gnp': "GrossNationalProduct",
                 'amc': 'American Motors Corporation', 'mpg': 'miles per gallon', 'asy': 'Ashley Municipal Airport',
                 'adh': 'Ardmore Downtown Executive Airport', 'ppt': 'PowerPoint', 'cd': 'compact disc',
                 'no': 'Number',
                 'US': 'United States', 'cv': 'Curriculum Vitae'}
    newTable = False
    new_table_tokens = []
    for tok in tokens:
        if tok.lower() in list(abbrevDic.keys()):
            if tok != "US":
                new_table_tokens.append(abbrevDic[tok.lower()])
                newTable = True
            else:
                new_table_tokens.append(abbrevDic[tok])
                newTable = True
        else:
            new_table_tokens.append(tok)

    if newTable:
        if is_camel_case(original):
            a = ("_").join(new_table_tokens)
            new_name = changeToCamel(a)
        elif is_snake_case(original):
            new_name = ("_").join(new_table_tokens)
        else:
            new_name = original
        print(f"{original} --> {new_name}")
        return new_name
    else:
        return None


def perturb_abbrevs(gt_schema):
    """
    Input: Gt_schema dict {'table': [col, col, col]}

    Output: perturbed_key {original: perturbed version}
    """
    perturbed_key = {}  # 'original': 'perTurbed'


    for table, column_list in gt_schema.items():
        #see if you can do table
        table_tokens = naiveTokenize(table)

        if table_tokens is not None:
            new_table_name = checkForAbbrev(table_tokens, table)
            if new_table_name is not None:
                perturbed_key[table] = new_table_name
            else:
                perturbed_key[table] = table

        else: #just one token, nothing to look up
            perturbed_key[table] = table


        for col in column_list:
            col_tokens = naiveTokenize(col)
            if col_tokens is not None:
                new_col_name = checkForAbbrev(col_tokens, col)
                if new_col_name is not None:
                    perturbed_key[col] = new_col_name
                else:
                    perturbed_key[col] = col
            else:
                perturbed_key[col] = col
    return perturbed_key


def add_table(db):
    """
    Pick a random table and its columns from a random DB. To be added to another DB.
    Input:
        db: String
    Output:
        random_db: String
        add_table_schema: Dict {table: [c1, c2, ...]}
    """
    add_table_schema = {}
    random_db, random_db_schema = get_random_db_schema(db)
    print(f"random db schema: {random_db_schema}")
    random_db_tables = list(random_db_schema.keys())
    if "sqlite_sequence" in random_db_tables:
        random_db_tables = [t for t in random_db_tables if t != "sqlite_sequence"]
    random_add_table = random.choice(random_db_tables)
    add_table_schema[random_add_table] = random_db_schema[random_add_table]
    return random_db, add_table_schema






#TODO: fix add_column
def get_schema_peturbations(tokens, gt_schema, random_db_schema):
    """
    Use: generate a bunch of possible schemas
    """
    schema_dict = {}

    inUseTablesList, inUseColumnsDict = get_inUse_tables_and_columns(gt_schema, tokens)

    def del_table(gt_schema, tokens):
        """
        Delete an unused table from the schema
        """
        del_table_schema = {}

        candidate_delete_tables = [k for k in list(gt_schema.keys()) if k not in tokens]
        delete_table = random.choice(candidate_delete_tables)
        pruned_tables = [t for t in gt_schema.keys() if t != delete_table]
        # fetch table info
        for table in pruned_tables:
            del_table_schema[table] = gt_schema[table]
        #print(f"del table schema: {del_table_schema}")
        return del_table_schema

    # def add_table(gt_schema, random_db_schema):
    #     """
    #     Add a random table from a random DB into the schema
    #     """
    #     add_table_schema = copy.deepcopy(gt_schema)
    #     random_db_tables = list(random_db_schema.keys())
    #     random_add_table = random.choice(random_db_tables)
    #     add_table_schema[random_add_table] = random_db_schema[random_add_table]
    #     return add_table_schema

    def del_column(gt_schema, tokens):
        del_col_schema = {}
        possible_tables_to_delete_from = [i for i in gt_schema.keys() if i not in tokens] #dont want to delete from a table in the query!
        table_to_delete_from = random.choice(possible_tables_to_delete_from)
        column_to_delete = random.choice(gt_schema[table_to_delete_from])
        for k in gt_schema.keys():
            if k != table_to_delete_from:
                del_col_schema[k] = gt_schema[k]
            else:
                cleaned_columns = [c for c in gt_schema[k] if c != column_to_delete]
                del_col_schema[k] = cleaned_columns

        #print(f"DEL COL SCHEMA: {del_col_schema}")
        return del_col_schema

    #TODO: FIX ME! Make sure that the random column you add isn't same name as an existing column in a table
    def add_column(gt_schema, random_db_schema):
        #TODO: add the column WHERE? To which table? for now, anywhere
        #TODO: make sure that the column youve chosen to add isn't already
        add_column_schema = copy.deepcopy(gt_schema)
        random_db_tables = list(random_db_schema.keys())
        random_db_table = random.choice(random_db_tables)
        column_to_add = random.choice(random_db_schema[random_db_table])

        gt_table_to_get_column = random.choice(list(gt_schema.keys()))
        add_column_schema[gt_table_to_get_column].append(column_to_add)
        return add_column_schema

    #TODO: address columns
    def add_punct(gt_schema, inUseTables, inUseColumns, tableOrColumn='table', inUse=True):
        new_schema = {}

        punct = random.choice(['%', '-', '.', '!', '?'])

        if inUse:
            table_to_perturb = random.choice(inUseTables)
        else:
            candidate_tables = [t for t in gt_schema.keys() if t not in inUseTables]
            table_to_perturb = random.choice(candidate_tables)

        if tableOrColumn == 'table':
            middlemark = int(len(table_to_perturb)/2)
            peturbed_table_name = f"`{table_to_perturb[:middlemark]}{punct}{table_to_perturb[middlemark:]}`"

            for t in gt_schema.keys():
                if t != table_to_perturb:
                    new_schema[t] = gt_schema[t]
                else:
                    new_schema[peturbed_table_name] = gt_schema[t]

        elif tableOrColumn == 'column':
            pass
            # candidate_columns = gt_schema[table_to_perturb]
            # #TODO: use in use columns
            # column_to_peturb = random.choice(candidate_columns)
            # middlemark = int(len(column_to_peturb) / 2)
            # peturbed_column_name = f"`{column_to_peturb[:middlemark]}{punct}{column_to_peturb[middlemark:]}`"
            # new_column_list = [c for c in candidate_columns if c!= column_to_peturb]
            # new_column_list.append(peturbed_column_name)
            # for t in gt_schema.keys():
            #     if t != table_to_perturb:
            #         new_schema[t] = gt_schema[t]
            #     else:
            #         new_schema[t] = new_column_list

        #print(f"NEW SCHEMA: {new_schema}")
        return new_schema

    def add_del_spaces(gt_schema, tokens, inUseTables, inUseColumns, tableOrColumn='table', inUse=True):
        pass




    schema_dict["del_table"] = del_table(gt_schema, tokens)
    schema_dict["add_table"] = add_table(gt_schema, random_db_schema)
    schema_dict["del_col"] = del_column(gt_schema, tokens)
    schema_dict["add_col"] = add_column(gt_schema, random_db_schema)
    schema_dict["add_punct_table_inUse"] = add_punct(gt_schema, inUseTablesList, inUseColumnsDict,
                                                     tableOrColumn='table', inUse=True)
    schema_dict["add_punct_table_notinUse"] = add_punct(gt_schema, inUseTablesList, inUseColumnsDict,
                                                        tableOrColumn='table', inUse=False)


    return schema_dict

def evaluate_single_query_with_perterbations(p, g_list, hardness=None, evaluator=None, scores=None,
                                             in_execution_order=False, verbose=False, peturb=True):

    schema_strategies = {}  # {"strategy": Result}

    for g in g_list:  # ground truth in grount truth list

        # step 0: get db info for getting schema
        g_str, db = g
        db_dir = "/export/home/sp-behavior-checking/data/spider/database"
        db_full_path = os.path.join(db_dir, db, db + ".sqlite")

        true_schema_graph = get_schema(db_full_path)
        print(f"TRUE DB SCHEMA: {true_schema_graph}")
        print(f"{type(true_schema_graph)}")

        # step 1: look in the sql to see what we can mess with
        # TODO: providing the schema isn't helping..
        tokens, token_types = mozsptokenize(p, value_tokenize=bu.tokenizer.tokenize,
                                            schema=true_schema_graph)
        print(f"tokens: {tokens}")
        print(f"token_types: {token_types}")


        #Step 0: get a RANDOM DB Schema for sampling in the db_peturbations
        rand_db_schema = get_random_db_schema(db)
        print(f"RANDOM DB: {rand_db_schema}")

        schema_dict = get_schema_peturbations(tokens, true_schema_graph, rand_db_schema)


        exit(12345)
        #print(f"schema dict: {schema_dict}")
    return schema_dict

def eval_prediction(pred, gt_list, dataset_id, db_name=None, in_execution_order=False, engine=None):
    if dataset_id == SPIDER:
        eval_dict = evaluate_single_query_with_perterbations(
            pred, [(gt, db_name) for gt in gt_list], in_execution_order=in_execution_order) #This returns a dict !!!!
        print(f"[eval_predition] eval_dict = {eval_dict}")
        return eval_dict
    else:
        raise NotImplementedError

def get_exact_match_metrics_peturb(examples, pred_list, in_execution_order=False, engine=None):
    assert(len(examples) == len(pred_list))

    peturbation_metrics = defaultdict(int)

    for i, example in enumerate(examples):
        top_k_preds = pred_list[i]
        # print(f"top_k_preds: {top_k_preds}") #1 str
        # print(type(top_k_preds)) #str
        em_recorded, ex_recorded = False, False
        # if isinstance(top_k_preds, list):
        #     for j, pred in enumerate(top_k_preds): #lol this is enumerating the string x)
        #         if example.dataset_id == SPIDER:
        #             gt_program_list = example.program_list
        #         elif example.dataset_id == WIKISQL:
        #             gt_program_list = example.program_ast_list_
        #         else:
        #             gt_program_list = example.gt_program_list
        #
        #         results = eval_prediction(pred, gt_program_list, example.dataset_id,
        #                                   db_name=example.db_name, in_execution_order=in_execution_order,
        #                                   engine=engine)
        #         #TODO: RESULTS IS A DICT HERE!!!!!
        if isinstance(top_k_preds, str):
            j=0
            pred = top_k_preds
            if example.dataset_id == SPIDER:
                gt_program_list = example.program_list
            elif example.dataset_id == WIKISQL:
                gt_program_list = example.program_ast_list_
            else:
                gt_program_list = example.gt_program_list
            print(f"{pred} ===> {gt_program_list[0]}")
            results = eval_prediction(pred, gt_program_list, example.dataset_id,
                                      db_name=example.db_name, in_execution_order=in_execution_order,
                                      engine=engine)

            print(f"RESULTS: {results}")
            # print(f"example.dataset_id: {example.dataset_id}")

            # if example.dataset_id == SPIDER:
            #     #em_correct = results[0]
            #     for label, eval in results.items():
            #         em_correct = eval[0]
            #         if em_correct:
            #             peturbation_metrics[label] +=1
        print(f"---------------------")

    for peturbation_type, num_correct in peturbation_metrics.items():
        print(f"PETURBATION {peturbation_type} = {float(num_correct / len(examples))}")

    return peturbation_metrics

################################### END DB PERTURBS. START NL PERTURBS ###################################
def fixIDtoken(tokens):
    new_tokens = []
    tokens = [str(t) for t in tokens]
    for i in range(len(tokens)):
        if tokens[i] == "i" and tokens[i+1] == "d":
            new_tokens.append("id")
        elif tokens[i] == "d" and tokens[i-1] == "i":
            dummy = "dummy"
        else:
            new_tokens.append(str(tokens[i]))
    return new_tokens

#using the original example tokens is messing with numbers
def reword_what_question(example, tokens):
    # valid_deps = [['attr', 'ROOT'], ['nsubj', 'ROOT']]
    basic_phrases = [['what', 'is'], ['what', 'are']]
    ask_verbs = ['give', 'tell', 'show', 'find', 'list', 'please']

    tokens = [str(t).lower() for t in tokens]

    if tokens[:2] in basic_phrases:
        base_tokens = tokens[2:]
        if base_tokens[-1] == '?':
            base_tokens = base_tokens[0:-1]
        decision = random.choice(['se', 'ask'])  # choose to ask like a search engine or with ask words like "give"
        if decision is 'se':
            new_nlq = (' ').join(base_tokens)
            return new_nlq
        elif decision is 'ask':
            new_tokens = []
            new_tokens.append(random.choice(ask_verbs))
            [new_tokens.append(t) for t in base_tokens]
            new_nlq = (' ').join(new_tokens)
            return new_nlq
    else:  # if its not a "what is/are" question, just add an ask verb (e.g. what animals --> show what animals)
        new_tokens = []
        new_tokens.append(random.choice(['show', 'give']))
        [new_tokens.append(t) for t in tokens]
        if new_tokens[-1] == '?':
            new_tokens = new_tokens[0:-1]

        new_nlq = (' ').join(new_tokens)
        return new_nlq

def reword_which_question(example, tokens, deps, pos):
    new_tokens = []
    valid_deps = [['det', 'nsubj'], ['det', 'amod'], ['det', 'npadvmod'], ['mark', 'amod'], ['det', 'compound']]
    if deps[0:2] in valid_deps:
        for i, tok in enumerate(tokens):
            if i >= 1:
                if deps[i] == 'ROOT' and pos[i] == 'AUX':
                    new_tokens.append("with")
                elif i == len(valid_deps) and str(tok) == '?':  # if its the last question mark
                    pass
                else:
                    new_tokens.append(str(tok).lower())

        new_nlq = (' ').join(new_tokens)
        return new_nlq
    else:
        # cannot reliably fudge with it
        new_tokens.append('show')
        [new_tokens.append(str(t).lower()) for t in tokens]
        if new_tokens[-1] == '?':
            new_tokens = new_tokens[0:-1]
        new_nlq = (' ').join(new_tokens)
        return new_nlq

def reword_who_question(example, deps, tokens):
    basic_phrases = [['who', 'is'], ['who', 'are']]
    tokens = [str(t).lower() for t in tokens]
    if tokens[:2] in basic_phrases and deps[:2] in [['nsubjpass', 'auxpass']]:
        if tokens[1] == 'is':
            new_tokens = ['the', 'person'] + tokens[2:]
        elif tokens[1] == 'are':
            new_tokens = ['the', 'people'] + tokens[2:]
        else:
            new_tokens = ['show'] + tokens  # dont know what to do

        new_nlq = (' ').join(new_tokens)
        return new_nlq
    elif tokens[:2] in basic_phrases and deps[:2] in [['nsubj', 'ROOT']]:  # you can just drop who is
        base_phrase = tokens[2:]
        new_nlq = (' ').join(base_phrase)
        return new_nlq
    else:
        new_tokens = ['show'] + tokens
        new_nlq = (' ').join(new_tokens)
        return new_nlq

#TODO: if new_nlq is 'show ...', pop off the "?" at the end
def replace_q_words(example, example_docobj):
    """
    Returns a new string to ask a question differently
    """
    #wh_tokens = ['who','what','when','where','which']
    #how_token = ['how']

    ask_verbs = ['give', 'tell', 'show', 'find', 'list', 'please']
    task_verbs = ['count', 'sort', 'compute']

    tokens = [token for token in example_docobj]
    pos = [token.pos_ for token in example_docobj]
    deps = [token.dep_ for token in example_docobj]

    start_token = str(tokens[0]).lower()



    if start_token == 'what':
        new_nlq = reword_what_question(example, tokens)
        return new_nlq

    if start_token == 'which':
        new_nlq = reword_which_question(example, tokens, deps, pos)
        return new_nlq

    if start_token == 'who':
        new_nlq = reword_who_question(example, deps, tokens)
        return new_nlq

    if start_token == 'where': #NB: only 1 'where' example in Spider dev
        new_tokens = ['show'] + [str(t).lower() for t in tokens]
        new_nlq = (' ').join(new_tokens)
        return new_nlq

    elif start_token == 'when':
        new_tokens = ['show'] + [str(t).lower() for t in tokens]
        new_nlq = (' ').join(new_tokens)
        return new_nlq

    elif start_token == 'how':
        new_tokens = ['show'] + [str(t).lower() for t in tokens]
        new_nlq = (' ').join(new_tokens)
        return new_nlq

    if start_token in ask_verbs: #can safely pop off the first token
        if start_token != 'please':
            new_tokens = [str(t).lower() for t in tokens[1:]]
        else:
            new_tokens = [str(t).lower() for t in tokens[2:]] #please X, delete the first two tokens
        new_nlq = (' ').join(new_tokens)
        return new_nlq
    elif start_token in task_verbs: #changing these kinds of verbs changes the query. Do not touch!
        return None
    else:
        print(f"[replace_q_words] UNKNOWN START TOKEN: {tokens[0]}. Query will be skipped.")
        return None

def synonym_non_schema(example, reserved):
    """
    Example obj
    Reserved is a list of tables+columns+picklist items.
    """
    new_tokens = []
    changed = False

    no_touchy = ["number", "name", "type", "id"] + sql_stopwords

    lemmatized_reserved = [lemmatizer.lemmatize(w) for w in reserved] + ['be']#lemmatize the reserved
    lemmatized_reserved = [w.lower() for w in lemmatized_reserved] #and lowercase, so there's no risk we modify these

    pos_tagged = nltk.pos_tag(word_tokenize(example.text))
    for word_tag in pos_tagged:
        word = word_tag[0].lower()
        tag = word_tag[1]
        if tag in ['NN', 'NNS'] and lemmatizer.lemmatize(word) not in lemmatized_reserved and word not in no_touchy and changed is False:
            word_sense = lesk(example.text, word_tag[0]) #returns a Synset. Not passing lowercase because maybe it helps with WSD
            try: #if the wordsense is not a NoneType
                all_candidate_lemmas = word_sense.lemma_names() #check possible lemmas
                pruned_candidate_lemmas = [l for l in all_candidate_lemmas if l != lemmatizer.lemmatize(word)] #dont choose the same thing
                if len(pruned_candidate_lemmas)>0:
                    new_word = random.choice(pruned_candidate_lemmas)
                    if tag == 'NN':
                        inflected_new_word = p.singular_noun(new_word)
                    else:
                        inflected_new_word = p.plural_noun(new_word)
                    if inflected_new_word is False: #inflect will return False if the word is already inflected correctly
                        new_tokens.append(new_word)# .upper())
                    else:
                        new_tokens.append(inflected_new_word) # .upper())
                    changed = True
                else: #there's no lemmas to choose, just keep the original word, and changed stays False
                    new_tokens.append(word_tag[0])
            except Exception as e_noWSD:
                #theres no lemmas to choose, just keep original word and changes stayes fales
                new_tokens.append(word_tag[0])
        else:
            new_tokens.append(word_tag[0])

    if changed:
        return (' ').join(new_tokens)
    else: #if nothing was changed, return None
        return None

def synonym_schema(example, reserved):
    """
    Example obj
    Reserved is a list of tables+columns+picklist items.

    Here we specifically modify reserved items
    """
    new_tokens = []
    changed = False

    no_touchy = ['number'] + sql_stopwords

    lemmatized_reserved = [lemmatizer.lemmatize(w) for w in reserved] + ['be']  # lemmatize the reserved
    lemmatized_reserved = [w.lower() for w in lemmatized_reserved]  # and lowercase, so there's no risk we modify these

    pos_tagged = nltk.pos_tag(word_tokenize(example.text))
    for word_tag in pos_tagged:
        word = word_tag[0].lower()
        tag = word_tag[1]
        if lemmatizer.lemmatize(word) in lemmatized_reserved and word not in no_touchy and changed is False:
            word_sense = lesk(example.text, word_tag[0])  # returns a Synset
            try:
                all_candidate_lemmas = word_sense.lemma_names()  # check possible lemmas
                pruned_candidate_lemmas = [l for l in all_candidate_lemmas if l != lemmatizer.lemmatize(word)]  # dont choose the same thing
                if len(pruned_candidate_lemmas) > 0:
                    new_word = random.choice(pruned_candidate_lemmas)
                    if tag == 'NN':
                        inflected_new_word = p.singular_noun(new_word)
                    else:
                        inflected_new_word = p.plural_noun(new_word)
                    if inflected_new_word is False:  # inflect will return False if the word is already inflected correctly
                        new_tokens.append(new_word) #.upper())
                    else:
                        new_tokens.append(inflected_new_word)  # .upper())
                    changed = True
                else:  # there's no lemmas to choose, just keep the original word, and changed stays False
                    new_tokens.append(word_tag[0])
            except Exception as e_noWSD:
                new_tokens.append(word_tag[0]) # there's no lemmas to choose, just keep the original word, and changed stays False
        else:
            new_tokens.append(word_tag[0]) #append the original

    if changed:
        return (' ').join(new_tokens)
    else:  # if nothing was changed, return None
        return None

def add_typo_non_schema(example, reserved, typoCorpusDict):
    """
    Example obj
    Reserved is a list of tables+columns+picklist items.
    typoCorpusDict has a list of typos for token keys

    Here we specifically do NOT modify reserved items
    """
    new_tokens = []
    changed = False
    no_touchy = ['mean', 'number', 'median'] + sql_stopwords

    lemmatized_reserved = [lemmatizer.lemmatize(w) for w in reserved] + ['be'] # lemmatize the reserved
    lemmatized_reserved = [w.lower() for w in lemmatized_reserved]  # and lowercase, so there's no risk we modify these

    pos_tagged = nltk.pos_tag(word_tokenize(example.text))
    for word_tag in pos_tagged:
        word = word_tag[0].lower()
        tag = word_tag[1]
        if tag in ['NN', 'NNS'] and lemmatizer.lemmatize(
                word) not in lemmatized_reserved and word not in no_touchy and changed is False:
            try:
                typo = random.choice(typoCorpusDict[word]) #doesnt need to be lemmatized
                new_tokens.append(typo)
                changed = True
            except Exception as e_dictMissingKey:
                new_tokens.append(word_tag[0])
        else:
            new_tokens.append(word_tag[0])
    if changed:
        return (' ').join(new_tokens)
    else:  # if nothing was changed, return None
        return None

def add_typo_schema(example, reserved, typoCorpusDict):
    """
    Example obj
    Reserved is a list of tables+columns+picklist items.
    typoCorpusDict has a list of typos for token keys

    Here we specifically modify reserved items
    """
    new_tokens = []
    changed = False

    lemmatized_reserved = [lemmatizer.lemmatize(w) for w in reserved]  + ['be'] # lemmatize the reserved, plus ignore modals
    lemmatized_reserved = [w.lower() for w in lemmatized_reserved]  # and lowercase, so there's no risk we modify these

    pos_tagged = nltk.pos_tag(word_tokenize(example.text))
    for word_tag in pos_tagged:
        word = word_tag[0].lower()
        tag = word_tag[1]
        if lemmatizer.lemmatize(word) in lemmatized_reserved and word not in sql_stopwords and changed is False:
            try:
                typo = random.choice(typoCorpusDict[word]) #doesnt need to be lemmatized
                new_tokens.append(typo)
                changed = True
            except Exception as e_dictMissingKey:
                new_tokens.append(word_tag[0])
        else:
            new_tokens.append(word_tag[0])
    if changed:
        return (' ').join(new_tokens)
    else:  # if nothing was changed, return None
        return None

def expandAbbrev(example, example_docobj):
    abbrevDic = {'id': 'identification', 'ids': 'identifications', 'ID': 'Identification', 'IDs': 'Identifications',
                 'tv': 'television', 'TV': 'television', 'WTA': "Womens' Tenis Association", 'GNP': "Gross National Product",
                 'amc': 'American Motors Corporation', 'mpg': 'miles per gallon', 'ASY': 'Ashley Municipal Airport',
                 'ADH': 'Ardmore Downtown Executive Airport', 'PPT': 'PowerPoint', 'CD': 'compact disc', 'No.': 'Number',
                 'US': 'United States', 'CV': 'Curriculum Vitae'}

    new_example = []
    changed = False
    tokens = [str(token) for token in example_docobj]

    if " id " in example.text:
        #force "id" to be ONE token, not two [i, d]. Thanks spacy.
        #this will mess with the dependency parse and POS, but oh well...
        tokens = fixIDtoken(tokens)
    for t in tokens:
        try:
            new_t = abbrevDic[t]
            new_example.append(new_t)
            changed = True
        except Exception as e:
            new_example.append(t)

    if changed:
        return (' ').join(new_example)
    else:
        return None

