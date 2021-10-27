"""
Usage: grammar.py DATASET DATAPATH MODE DB [options]

Run a simple SCFG grammar given an input DB

Arguments:
    DATASET                String name of dataset [default: spider]
    DATAPATH               String path to dir containing tables.json [default: data/spider]
    MODE                   String status "all" or "one" for running tests on all db's or only one specified in arg DB below [default: one]
    DB                     String name of DB to work with (e.g. "allergy_1") or "random" to randomly choose [default: random]

Options:
  --debug                  Enable debug-level logging.

#Unsupported features to implement:

Example:
  python grammar.py spider data/spider all random <----- this one to generate them all!
  python grammar.py spider data/spider one random
  python grammar.py spider data/spider one allergy_1
  python grammar.py spider data/spider one sakila_1
"""

from docopt import docopt
import logging
import logzero
from logzero import logger as log
import random
import time
import string
from collections import defaultdict
import json
import os
from pathlib import Path

from src.data_processor.schema_loader import load_schema_graphs_spider

######## Global SCFG variables ########
q1 = 'Select'
w_all = 'all columns'
w_distinct = 'unique'
w_from = 'from' #of
w_where = 'when'
w_and = 'and'
w_equals = 'equals'
w_max = 'maximum'
w_min = 'minimum'
w_sum = 'the sum of'
w_count = 'the number of'
w_avg = 'the average value of'
w_orderby = random.choice(['ordered by', 'sorted by' ])
w_groupby = 'grouped by'
w_asc = 'in ascending order'
w_desc = 'in descending order'
w_like_start = 'starts with'
w_like_end= 'ends with'
w_not_like_start = 'does not start with'
w_not_like_end = 'does not end with'
w_having = 'with' #'having'
w_between = 'is between'
w_exists = 'there is a result for'
w_not_in = 'is not in'
w_is_in = 'is in'

#skip any tables or coluns with this name
sql_stopwords = ['and', 'as', 'asc', 'between', 'case', 'collate_nocase', 'cross_join', 'desc', 'else', 'end', 'from',
                 'full_join', 'full_outer_join', 'group_by', 'having', 'in', 'inner_join', 'is', 'is_not', 'join',
                 'left_join', 'left_outer_join', 'like', 'limit', 'none', 'not_between', 'not_in', 'not_like', 'offset',
                 'on', 'or', 'order_by', 'reserved', 'right_join', 'right_outer_join', 'select', 'then', 'union',
                 'union_all', 'except', 'intersect', 'using', 'when', 'where', 'binary_ops', 'unary_ops', 'with',
                 'durations', 'max', 'min', 'count', 'sum', 'avg']

random.seed(27)


#######################################
def aliasTable2TrueTableLUT(db):
    alias2TrueTableDict = {} #badalias: (truealias, true_table)
    """
     'get_table', 
    'get_table_by_name', 'get_table_id', 'get_table_scopes', 'index_table', 'indexed_table', 
     'is_table_name',
       'node_index', 'node_rev_index',
       'table_index', 
       'table_names', 'table_rev_index']
       """
    for alias, i in db.table_index.items():
        true_table_name = db.node_rev_index[i].name
        true_alias = db.node_rev_index[i].normalized_name
        alias2TrueTableDict[alias] = (true_alias, true_table_name)

    return alias2TrueTableDict

def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]

def badCol(column_name):
    for p in string.punctuation:
        if column_name.endswith(p) or column_name.startswith(p):
            return True
    return False

def getColAlias(db, table_name, true_col_name):
    #print(f"db: {db}, table_name: {table_name}, true_col_name: {true_col_name}")
    table1_index = db.table_index[table_name]  # get the int for the table name

    unflattened_relevant_column_indexs = db.get_current_schema_layout(
        tables=[table1_index])  # [table1_index]  # get the field ints for the table
    relevant_column_indexs = [item for sublist in unflattened_relevant_column_indexs for item in sublist]

    for colinx, field_info in zip(relevant_column_indexs, db.get_table(table1_index).fields):
        col = db.get_field(colinx)
        if true_col_name == col.name:
            return col.normalized_name
    return true_col_name


def getColsAndTypes(db, table_name):
    #print(f"table name: {table_name}")
    table1_index = db.table_index[table_name]  # get the int for the table name
    #print(f"table1_index: {table1_index}")
    unflattened_relevant_column_indexs = db.get_current_schema_layout(tables=[table1_index]) # [table1_index]  # get the field ints for the table
    relevant_column_indexs = [item for sublist in unflattened_relevant_column_indexs for item in sublist]

    column_candidates = [] #cleaned column names for relevant table
    keep_column_indexes = [] #list of ints refering to column indexs. Will usually be the same as relevant_column_indexes
    keep_column_types = []

    for colinx, field_info in zip(relevant_column_indexs, db.get_table(table1_index).fields):
        col = db.get_field(colinx)
        #print(col.normalized_name) alias name
        column_name = col.name
        data_type = field_info.data_type
        if " " not in column_name and column_name.lower() not in sql_stopwords and badCol(column_name) is False:
            #skip problematic columns like 'Home Town' that break official Spider evaluation script.
            column_candidates.append(column_name)
            keep_column_indexes.append(colinx)
            keep_column_types.append(data_type)

    assert len(column_candidates) == len(keep_column_indexes) #each candidate column should have an associated type, or something is wrong
    assert len(column_candidates) == len(keep_column_types)
    return column_candidates, keep_column_types, keep_column_indexes

def getColumnIndex(db, table_name, column_name):
    """
    For when you need to get the column index of a specific column for a table.
    Column indexes are needed for getting the picklist, for example
    """
    table1_index = db.table_index[table_name]  # get the int for the table name
    relevant_column_indexs = db.get_current_schema_layout()[table1_index]

    col_index_dict = {} #columnName: 0

    for colinx in relevant_column_indexs:
        col = db.get_field(colinx)
        col_name = col.name
        #if " " not in column_name: #force ignore illformatted column names
        col_index_dict[col_name] = colinx

    specified_column_index = col_index_dict[column_name]

    return specified_column_index

def precompute_compatibility(schema_graphs):
    """
    Input: schema_graphs <SchemaGraphs> object with all loaded DB's;
    Output: compatibility_dict <Dict>
    {'db_name1':
            {
            'table_name1':
                {'numColumns': Int,
                'numberColumnsTypeNumber': Int,
                'numberColumnsTypeText': Int,
                'picklistProblem': Bool},
            'table_name2': {dict about columns},
            'table_name3': {dict about columns}
            }
    'db_name2': {dict of tables},
    'db_name3': {dict of tables},
    ...
    'db_nameN': {dict of tables}
    }
    """
    compatibility_dict = {}
    for db_name, db_id in list(schema_graphs.db_index.items()):
        db = schema_graphs[db_name]

        list_of_tables = list(db.table_names)
        table_dict = {}
        for table_name in list_of_tables:

            columns, coltypes, rel_indexs = getColsAndTypes(db, table_name) #true column names, not signatures!

            numColumns = len(columns)
            numberColumnsTypeNumber = coltypes.count("number")
            numberColumnsTypeText = coltypes.count("text")

            #check for picklist problems
            list_of_picklists = [db.get_field_picklist(i) for i in rel_indexs]
            num_bad_picklists = 0

            for pl in list_of_picklists:
                try:
                    random.choice(pl)
                except Exception as e:
                    num_bad_picklists += 1

            if num_bad_picklists >= 1:
                picklistProblem = True
            else:
                picklistProblem = False

            column_info_dict = {'numColumns': numColumns, 'numberColumnsTypeNumber': numberColumnsTypeNumber,
                               'numberColumnsTypeText': numberColumnsTypeText, 'picklistProblem': picklistProblem}
            table_dict[table_name] = column_info_dict

            if table_name == list_of_tables[-1]:
                compatibility_dict[db_name] = table_dict

    return compatibility_dict

def get_table_k_columns(db, table=None, k=1, type_constraint=None):
    """
    Input: db <SchemaGraph object> ;
           table <String> you can give the name of a specific table, otherwise a random one is chosen [default: None,
           a random one is chosen] ;
           k <Int> number of columns [default 1] ;
           type_constraint <Str> [options: None, "text", "number", "time", "boolean"]
    Output: table1 <Str> ;
            columns: List of Strings if k>2, otherwise if k=1 just a String.
            If a List of Strings for multiple columns is returned here, then string formatting must be handled in the
            method where this function is called.
    """
    original_k = k
    if table is None:
        table1 = random.choice(list(db.table_names))  # choose a random table
    else:
        table1 = table

    column_candidates, column_types, relevant_indexs = getColsAndTypes(db, table1)

    #prune column candidate by specified type constraint
    if type_constraint is None:
        filtered_column_candidates = column_candidates
    else:
        filtered_column_candidates = []
        for col, type in zip(column_candidates, column_types):
            if type == type_constraint:
                filtered_column_candidates.append(col)

        if len(filtered_column_candidates) == 0:
            print(f"uh oh! there were no columns in table {table1} of specified type {type_constraint}...")
            print(f"support for this exception needs to be added. Exit code: 111")
            exit(111)

    if k == 1:
        kcols = random.choice(filtered_column_candidates)  #string
        #print(f"one col: {kcols}")
    else:
        #A check that k cannot be larger than len(column_candidates)
        if k > len(filtered_column_candidates):
            k = len(filtered_column_candidates) #change the length of k if k is too big

        kcols = random.sample(filtered_column_candidates, k=k) #list of strings
        #print(f"kcols: {kcols}")
        if len(kcols) < original_k:
            print(f"* ALERT!! the number of selected columns ({len(kcols)}) is less than what you asked for ({original_k})")

    return table1, kcols

def is_stupid_var(var):
    """
    Return True if the variable is problematic and needs re-formatting
    Return False if the variable is fine the way it is
    """
    if not isinstance(var, str): #if the var isnt a string, its fine.
        return False

    del_puncts = [p for p in string.punctuation if p != ','] #commas seem harmless, no?

    if "'" in var:
        return True
    elif '\n' in var:
        return True
    elif 'None' in var:
        return True
    else:
        for p in del_puncts:  # if there's any punctuation at all, its probably wrecked
            if p in var:
                return True
    try:
        var.encode('ascii') #unicode characters break a lot of Spider eval scripts
    except Exception:
        return True

    return False

def get_column_value(db, table, column):
    """
    Chooses 1 value in the provided column for the given table and db.
    Input: db <SchemaGraph>, table <String>, column <String>
    Output: value <String>
    """
    #This potentially also needs foreign key, primary key info though?
    column_index = getColumnIndex(db, table, column)
    picklist = db.get_field_picklist(column_index)
    value = str(random.choice(picklist))
    return value

#TODO: add a thing that forces the values to be the right datatype
def get_multiple_column_value(db, table, column, k=2):
    """
    Chooses 1 value in the provided column for the given table and db.
    Input: db <SchemaGraph>, table <String>, column <String>, k <Int> values
    Output: value list <List>
    """
    #This potentially also needs foreign key, primary key info though?
    column_index = getColumnIndex(db, table, column)
    picklist = db.get_field_picklist(column_index)
    print(f"picklist: {picklist}")
    max_num_k = len(picklist)
    if k > max_num_k: #reset k to the number of things in the picklist, if k > length of picklist!
        print(f"* ALERT!! the number of column values selected ({max_num_k}) is less than what you asked for ({k})")
        k = max_num_k
    value_list = random.sample(picklist, k=k)
    print(f"value_list: {value_list}")
    print(f"{[type(v) for v in value_list]}")
    return value_list

def check_unique_column(db, table, column):
    all_tables = list(db.table_names)
    other_tables = [t for t in all_tables if t is not table]

    other_columns = set()

    for ot in other_tables:
        cols, col_types, col_indexs = getColsAndTypes(db, ot)
        [other_columns.add(c) for c in cols]

    # print(f"other columns: {other_columns}")
    # print(f"selected column: {column}")
    if isinstance(column, str):
        if column in other_columns:
            return False #False, column is not unique
        else:
            return True
    elif isinstance(column, list):
        bools = []
        for col in other_columns:
            if col in other_columns:
                bools.append(False)
            else:
                bools.apply(True)
        if False in bools:
            return False
        else:
            return True

############ SQL CONDITIONS ################
#to be used in WHERE clauses


def and1(db, table1, columns):
    condition = 'and'

    list_of_columns_as_strings = []
    list_of_columns_as_sql = []

    operator = 'equals'
    symbol = '='

    for i, c in enumerate(columns):
        ac = getColAlias(db, table1, c)
        val = get_column_value(db, table1, c)

        # check to see if the value is problematic or not:
        if is_stupid_var(val):
            return "BADVAL", "BADVAL"  # needs 2 objects to unpack"

        try:
            float(val)
            formatted_sql_var = f"{val}"
        except Exception as e:
            formatted_sql_var = f"'{val}'"

        if i == 0:
            col_as_string = f"{ac} {operator} {val}"
            col_as_sql = f"{c} {symbol} {formatted_sql_var}"
        else:
            col_as_string = f"{condition} {ac} {operator} {val}"
            col_as_sql = f"{condition.upper()} {c} {symbol} {formatted_sql_var}"

        list_of_columns_as_strings.append(col_as_string)
        list_of_columns_as_sql.append(col_as_sql)

    formatted_where_columns_with_vars = ' '.join(list_of_columns_as_strings)
    formatted_where_columns_as_sql = ' '.join(list_of_columns_as_sql)

    return formatted_where_columns_with_vars, formatted_where_columns_as_sql


def or1(db, table1, columns):
    condition = 'or'

    list_of_columns_as_strings = []
    list_of_columns_as_sql = []

    operator = 'equals'
    symbol = '='

    for i, c in enumerate(columns):
        ac = getColAlias(db, table1, c)
        val = get_column_value(db, table1, c)

        # check to see if the value is problematic or not:
        if is_stupid_var(val):
            return "BADVAL", "BADVAL"  # needs 2 objects to unpack"

        try:
            float(val)
            formatted_sql_var = f"{val}"
        except Exception as e:
            formatted_sql_var = f"'{val}'"

        if i == 0:
            col_as_string = f"{ac} {operator} {val}"
            col_as_sql = f"{c} {symbol} {formatted_sql_var}"
        else:
            col_as_string = f"{condition} {ac} {operator} {val}"
            col_as_sql = f"{condition.upper()} {c} {symbol} {formatted_sql_var}"

        list_of_columns_as_strings.append(col_as_string)
        list_of_columns_as_sql.append(col_as_sql)

    formatted_where_columns_with_vars = ' '.join(list_of_columns_as_strings)
    formatted_where_columns_as_sql = ' '.join(list_of_columns_as_sql)

    return formatted_where_columns_with_vars, formatted_where_columns_as_sql



#Can mix ANDs and ORs
def andor2(db, table1, columns):

    list_of_columns_as_strings = []
    list_of_columns_as_sql = []

    operator = 'equals'
    symbol = '='

    for i, c in enumerate(columns):
        condition = random.choice(['and', 'or'])
        val = get_column_value(db, table1, c)
        ac = getColAlias(db, table1, c)

        if is_stupid_var(val):
            return "BADVAL", "BADVAL" #needs 2 objects to unpack"
            # scrubbed_val = re.sub('\n', '', val)
            # formatted_sql_var = f"`'{scrubbed_val}'`"  # if the var is problematic, add escape characters
        try:
            float(val)
            formatted_sql_var = f"{val}"
        except Exception as e:
            formatted_sql_var = f"'{val}'"

        if i == 0:
            col_as_string = f"{ac} {operator} {val}"
            col_as_sql = f"{c} {symbol} {formatted_sql_var}"
        else:
            col_as_string = f"{condition} {ac} {operator} {val}"
            col_as_sql = f"{condition.upper()} {c} {symbol} {formatted_sql_var}"

        list_of_columns_as_strings.append(col_as_string)
        list_of_columns_as_sql.append(col_as_sql)

    formatted_columns_with_vars = ' '.join(list_of_columns_as_strings)
    formatted_columns_as_sql = ' '.join(list_of_columns_as_sql)

    return formatted_columns_with_vars, formatted_columns_as_sql

#Requires "text" column type
def like1(db, table1, col, value):
    starts_or_ends = random.choice([w_like_start, w_like_end])
    acol = getColAlias(db, table1, col)

    try:
        if starts_or_ends is w_like_start:
            letter = value[0]
            formatted_letter = f"%{letter}"
        else:
            letter = value[-1]
            formatted_letter = f"{letter}%"
    except Exception as e:
        return "BADVAL", "BADVAL"

    if letter.lower() not in string.ascii_lowercase:
        return "BADVAL", "BADVAL"

    nlq_end = f"{acol} {starts_or_ends} {letter} "
    sql_end = f"{col} LIKE '{formatted_letter}'"
    return nlq_end, sql_end


def notlike1(db, table1, col, value):
    starts_or_ends = random.choice([w_not_like_start, w_not_like_end])
    acol = getColAlias(db, table1, col)

    try:
        if starts_or_ends is w_like_start:
            letter = value[0]
            formatted_letter = f"%{letter}"
        else:
            letter = value[-1]
            formatted_letter = f"{letter}%"
    except Exception as e:
        return "BADVAL", "BADVAL"


    if letter.lower() not in string.ascii_lowercase:
        return "BADVAL", "BADVAL"

    nlq_end = f"{acol} {starts_or_ends} {letter} "
    sql_end = f"{col} NOT LIKE '{formatted_letter}'"
    return nlq_end, sql_end


def between1(db, table1, column):
    #get two random numeric values from the column
    values = sorted(get_multiple_column_value(db, table1, column, k=2))
    aliascol = getColAlias(db, table1, column)

    try:
        float(values[0])
        float(values[1])
    except Exception as e:
        return "BADVAL", "BADVAL"

    nlq_end = f"{aliascol} {w_between} {values[0]} {w_and} {values[1]}"
    sql_end = f"{column} BETWEEN {values[0]} AND {values[1]}"
    return nlq_end, sql_end

def wherenot3(db, col1, columns, table=None, tableLUT=None):
    candidate_cols = [col for col in columns if col != col1]
    col2 = random.choice(candidate_cols)
    aliascol2 = getColAlias(db, table, col2)
    nl_part, sql_part = select_col(db, table=table, tableLUT=tableLUT) #use most simple
    sql_part = sql_part.strip(";")
    nlq_end = f"{aliascol2} {w_not_in} \"{nl_part}\" "
    sql_end = f"{col2} NOT IN ({sql_part})"
    return nlq_end, sql_end


def wherein(db, col1, columns, table=None, tableLUT=None):
    candidate_cols = [col for col in columns if col != col1]
    col2 = random.choice(candidate_cols)
    aliascol2 = getColAlias(db, table, col2)
    nl_part, sql_part = select_col(db, table=table, tableLUT=tableLUT) #use most simple
    sql_part = sql_part.strip(";")
    nlq_end = f"{aliascol2} {w_is_in} \"{nl_part}\" "
    sql_end = f"{col2} IN ({sql_part})"
    return nlq_end, sql_end

def whereMaths(db, table=None, tableLUT=None):
    table1, columns = get_table_k_columns(db, k=2, table=table, type_constraint="number")
    if isinstance(columns, str):
        return "BADVAL", "BADVAL"
    # if len(columns) == 1:
    #     return "BADVAL", "BADVAL"

    print(f"COLUMNS: {columns}")
    col1 = columns[0]
    col2 = columns[1]
    val2 = get_column_value(db, table1, col2)

    alias_col1 = getColAlias(db, table, col1)
    alias_col2 = getColAlias(db, table, col2)

    operatordict = {"less than": "<", "greater than": ">", "less than or equal to": "<=",
                    "greater than or equal to": ">="}
    # <> breaks the sql parser :'(

    operator, symbol = random.choice(list(operatordict.items()))

    try:
        float(val2)
    except Exception as e:  # force it to non numerical for string values! can't be >= elephant.
        return "BADVAL", "BADVAL"

    if is_stupid_var(val2):
        return "BADVAL", "BADVAL" #needs 2 objects to unpack"
        # scrubbed_val2 = re.sub('\n', '', val2)
        # formatted_sql_var = f"`'{scrubbed_val2}'`"  # if the var is problematic, add escape characters
    try:
        float(val2)
        formatted_sql_var = f"{val2}"
    except Exception as e:
        return "BADVAL", "BADVAL"

    decision = random.choice([(w_all, '*'), (alias_col1, col1)])  # decide to select all or select just the 1 column

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    nlq = f"{q1} {decision[0]} {w_from} {gt_alias_table} {w_where} {alias_col2} is {operator} {val2}"
    sql = f"SELECT {decision[1]} FROM {gt_table} WHERE {col2} {symbol} {formatted_sql_var} ;"
    print("*****" * 10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****" * 10)
    return nlq, sql


######################## SQL CLAUSES ########################
def select_col(db, table=None, tableLUT=None):
    """
    Same as select1, except you cannot select all
    """
    table1, col1 = get_table_k_columns(db, table)
    aliascol = getColAlias(db, table1, col1)

    decision = (aliascol, col1)  # decide to select all or select just the 1 column

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    # if decision is col1:  # check if its a unique column
    #     unique_column = check_unique_column(db, table1, col1)
    #     if unique_column:
    #         nlq = f"{col1}"
    #         sql = f"{col1} FROM {table1} ;"
    # else:  # if the column names are not unique to the table in the DB, then we need a FROM table
    nlq = f"{decision[0]} {w_from} {gt_alias_table}"
    sql = f"SELECT {decision[1]} FROM {gt_table} ;"
    print("*****" * 10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****" * 10)
    return nlq, sql


def select1(db, table=None, tableLUT=None):
    """
    Select either column1 or all columns from table1

    Input: db <SchemaGraph object>
    Output: String: natural language question; String: sql
    """
    table1, col1 = get_table_k_columns(db, table)
    alias_col1 = getColAlias(db, table1, col1)

    decision = random.choice([(w_all, '*'), (alias_col1, col1)]) #decide to select all or select just the 1 column

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    # if decision is col1: #check if its a unique column
    #     unique_column = check_unique_column(db, table1, col1)
    #     if unique_column:
    #         nlq = f"{q1} {col1}"
    #         sql = f"SELECT {col1} FROM {table1} ;"
    # else: #if the column names are not unique to the table in the DB, then we need a FROM table

    nlq = f"{q1} {decision[0]} {w_from} {gt_alias_table}"
    sql = f"SELECT {decision[1]} FROM {gt_table} ;"
    print("*****"*10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****" * 10)
    return nlq, sql

def selectk(db, k, table=None, tableLUT=None):
    """
    Select multiple columns from table1

    Input: db <SchemaGraph object>, k <Int> number of desired columns
    Output: String: natural language question; String: sql
    """
    table1, columns = get_table_k_columns(db, k=k, table=table)

    print(f"columns: {columns}")
    alias_columns = [getColAlias(db, table, c) for c in columns]
    print(f"alias columns: {alias_columns}")
    alias_columns_as_string = ', '.join(alias_columns)

    columns_as_string = ', '.join(columns)
    #unique = check_unique_column(db, table, columns)

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    nlq = f"{q1} {alias_columns_as_string} {w_from} {gt_alias_table}"
    sql = f"SELECT {columns_as_string} FROM {gt_table} ;"
    print("*****"*10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****" * 10)
    return nlq, sql

def distinct1(db, table=None, tableLUT=None):
    table1, col1 = get_table_k_columns(db, table=table)
    aliascol = getColAlias(db, table1, col1)

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    nlq = f"{q1} {w_distinct} {aliascol} {w_from} {gt_alias_table}"
    sql = f"SELECT DISTINCT {col1} FROM {gt_table} ;"
    print("*****"*10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****"*10)
    return nlq, sql


def where1(db, table=None, tableLUT=None):
    table1, columns = get_table_k_columns(db, k=2, table=table)
    col1 = columns[0]
    col2 = columns[1]
    val2 = get_column_value(db, table1, col2)

    aliascol1 = getColAlias(db, table1, col1)
    aliascol2 = getColAlias(db, table1, col2)

    if is_stupid_var(val2):
        return "BADVAL", "BADVAL" #needs 2 objects to unpack"
        # scrubbed_val2 = re.sub('\n', '', val2)
        # formatted_sql_var = f"`'{scrubbed_val2}'`"  #= if the var is problematic, add escape characters
    else:
        formatted_sql_var = f"'{val2}'"

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    try:
        aliascol1.encode('ascii')
        aliascol2.encode('ascii')
        val2.encode('ascii')
    except UnicodeEncodeError:
        print("hi")
        print(aliascol1)
        print(aliascol2)
        print(val2)


    nlq = f"{q1} {aliascol1} {w_from} {gt_alias_table} {w_where} {aliascol2} {w_equals} {val2}"
    sql = f"SELECT {col1} FROM {gt_table} WHERE {col2} = {formatted_sql_var};"
    print("*****"*10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****"*10)
    return nlq, sql

def where2(db, table=None, tableLUT=None):
    table1, columns = get_table_k_columns(db, k=2, table=table)
    col1 = columns[0]
    col2 = columns[1]
    val2 = get_column_value(db, table1, col2)

    aliascol1 = getColAlias(db, table1, col1)
    aliascol2 = getColAlias(db, table1, col2)

    operator = "equals"
    symbol = "="

    if is_stupid_var(val2):
        return "BADVAL", "BADVAL" #needs 2 objects to unpack"

    try:
        float(val2)
        formatted_sql_var = f"{val2}" #dont put quotes around numbers
    except Exception as e:
        formatted_sql_var = f"'{val2}'"

    try:
        float(val2)
    except Exception as e:
        return "BADVAL", "BADVAL"

    decision = random.choice([(w_all, '*'), (aliascol1, col1)])  # decide to select all or select just the 1 column

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    nlq = f"{q1} {decision[0]} {w_from} {gt_alias_table} {w_where} {aliascol2} {operator} {val2}"
    sql = f"SELECT {decision[1]} FROM {gt_table} WHERE {col2} {symbol} {formatted_sql_var};"
    print("*****" * 10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****" * 10)
    return nlq, sql


def whereNotEquals(db, table=None, tableLUT=None):
    table1, columns = get_table_k_columns(db, k=2, table=table)
    col1 = columns[0]
    col2 = columns[1]
    val2 = get_column_value(db, table1, col2)

    operator = "does not equal"
    symbol = "!="

    aliascol1 = getColAlias(db, table1, col1)
    aliascol2 = getColAlias(db, table1, col2)

    if is_stupid_var(val2):
        return "BADVAL", "BADVAL" #needs 2 objects to unpack"

    try:
        float(val2)
        formatted_sql_var = f"{val2}" #dont put quotes around numbers
    except Exception as e:
        formatted_sql_var = f"'{val2}'"

    try:
        float(val2)
    except Exception as e:
        return "BADVAL", "BADVAL"

    decision = random.choice([(w_all, '*'), (aliascol1, col1)])  # decide to select all or select just the 1 column

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    nlq = f"{q1} {decision[0]} {w_from} {gt_alias_table} {w_where} {aliascol2} {operator} {val2}"
    sql = f"SELECT {decision[1]} FROM {gt_table} WHERE {col2} {symbol} {formatted_sql_var};"
    print("*****" * 10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****" * 10)
    return nlq, sql

#doesnt matter what datatype of column is. Just needs minimum 2 columns
def whereNested1(db, k=3, table=None, tableLUT=None):
    nested_where_results_list = [] #list of triples tuple(nlq, sql, "str rule name")

    table1, columns = get_table_k_columns(db, k=k, table=table)
    immutable_table1 = table1 #never change this!!!! for the recursion
    col1 = columns[0]
    aliascol1 = getColAlias(db, table1, col1)

    decision = random.choice([(w_all, '*'), (aliascol1, col1)])  # decide to select all or select just the 1 column

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1


    base_nlq = f"{q1} {decision[0]} {w_from} {gt_alias_table} {w_where}"
    base_sql = f"SELECT {decision[1]} FROM {gt_table} WHERE"

    nlq_ending, sql_ending = and1(db, immutable_table1, columns)
    if nlq_ending != "BADVAL":
        nlq = f"{base_nlq} {nlq_ending}"
        sql = f"{base_sql} {sql_ending} ;"
        print("*****" * 10)
        print(f"/ {nlq}")
        print(f'\\ {sql}')
        print("*****" * 10)
        nested_where_results_list.append((nlq, sql, "and"))

    nlq_ending, sql_ending = or1(db, immutable_table1, columns)
    if nlq_ending != "BADVAL":
        nlq = f"{base_nlq} {nlq_ending}"
        sql = f"{base_sql} {sql_ending} ;"
        print("*****" * 10)
        print(f"/ {nlq}")
        print(f'\\ {sql}')
        print("*****" * 10)
        nested_where_results_list.append((nlq, sql, "or"))


    nlq_ending, sql_ending = andor2(db, immutable_table1, columns)
    if nlq_ending != "BADVAL":
        nlq = f"{base_nlq} {nlq_ending}"
        sql = f"{base_sql} {sql_ending} ;"
        print("*****" * 10)
        print(f"/ {nlq}")
        print(f'\\ {sql}')
        print("*****" * 10)
        nested_where_results_list.append((nlq, sql, "andandor"))

    nlq_ending, sql_ending = wherenot3(db, col1, columns, table=immutable_table1, tableLUT=tableLUT)
    nlq = f"{base_nlq} {nlq_ending}"
    sql = f"{base_sql} {sql_ending} ;"
    print("*****" * 10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****" * 10)
    nested_where_results_list.append((nlq, sql, "not"))

    nlq_ending, sql_ending = wherein(db, col1, columns, table=immutable_table1, tableLUT=tableLUT)
    nlq = f"{base_nlq} {nlq_ending}"
    sql = f"{base_sql} {sql_ending} ;"
    print("*****" * 10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****" * 10)
    nested_where_results_list.append((nlq, sql, "in"))

    count_per_method = len(nested_where_results_list)
    return nested_where_results_list, count_per_method

#requires at least two "text" type columns!
def whereNested2(db, k=2, table=None, tableLUT=None):
    nested_where_results_list = []  # list of triples tuple(nlq, sql, "str rule name")

    table1, columns = get_table_k_columns(db, k=k, table=table, type_constraint="text")
    col1 = columns[0]
    col2 = columns[1]
    val2 = get_column_value(db, table1, col2)

    aliascol1 = getColAlias(db, table1, col1)

    decision = random.choice([(w_all, '*'), (aliascol1, col1)])  # decide to select all or select just the 1 column

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
            lookup = tableLUT[table1]
            gt_alias_table = lookup[0]
            gt_table = lookup[1]
        else:
            gt_alias_table = table1
            gt_table = table1

    base_nlq = f"{q1} {decision[0]} {w_from} {gt_alias_table} {w_where}"
    base_sql = f"SELECT {decision[1]} FROM {gt_table} WHERE"

    nlq_end, sql_end = like1(db, table1, col2, val2) #this one shouldn't have any problem with "stupid vars"
    if nlq_end != "BADVAL":
        nlq = f"{base_nlq} {nlq_end}"
        sql = f"{base_sql} {sql_end} ;"
        print("*****" * 10)
        print(f"/ {nlq}")
        print(f'\\ {sql}')
        print("*****" * 10)
        nested_where_results_list.append((nlq, sql, "like"))
    else:
        print("LIKE was bad")

    nlq_end, sql_end = notlike1(db, table1, col2, val2) #this one shouldn't have any problem with "stupid vars"
    if nlq_end != "BADVAL":
        nlq = f"{base_nlq} {nlq_end}"
        sql = f"{base_sql} {sql_end} ;"
        print("*****" * 10)
        print(f"/ {nlq}")
        print(f'\\ {sql}')
        print("*****" * 10)
        nested_where_results_list.append((nlq, sql, "notlike"))
    else:
        print("LIKE was bad")

    count_per_method = len(nested_where_results_list)
    return nested_where_results_list, count_per_method

#requires at least two "number" type columns!
def whereNested3(db, k=3, table=None, tableLUT=None):
    table1, columns = get_table_k_columns(db, k=k, table=table, type_constraint="number")
    col1 = columns[0]
    immutable_table1 = table1

    aliascol1 = getColAlias(db, table1, col1)

    decision = random.choice([(w_all, '*'), (aliascol1, col1)])  # decide to select all or select just the 1 column

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    base_nlq = f"{q1} {decision[0]} {w_from} {gt_alias_table} {w_where}"
    base_sql = f"SELECT {decision[1]} FROM {gt_table} WHERE"

    nlq_end, sql_end = between1(db, immutable_table1, random.choice(columns[1:]))
    if nlq_end != "BADVAL":
        nlq = f"{base_nlq} {nlq_end}"
        sql = f"{base_sql} {sql_end} ;"
        print("*****" * 10)
        print(f"/ {nlq}")
        print(f'\\ {sql}')
        print("*****" * 10)
        return nlq, sql
    else:
        return "BADVAL", "BADVAL"

def orderby1(db, table=None, tableLUT=None): #no ascending/descending specifications, and no field/column type_constraint
    table1, columns = get_table_k_columns(db, k=2, table=table)
    col1 = columns[0]
    col2 = columns[1]
    aliascol1 = getColAlias(db, table1, col1)
    aliascol2 = getColAlias(db, table1, col2)

    decision = random.choice([(w_all, '*'), (aliascol1, col1)])  # decide to select all or select just the 1 column

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    nlq = f"{q1} {decision[0]} {w_from} {gt_alias_table} {w_orderby} {aliascol2} "
    sql = f"SELECT {decision[1]} FROM {gt_table} ORDER BY {col2};"
    print("*****"*10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****"*10)
    return nlq, sql

def orderby2(db, table=None, tableLUT=None): #with ascending conditioned on NUM type column
    table1, columns = get_table_k_columns(db, k=2, type_constraint="number", table=table)
    col1 = columns[0]
    col2 = columns[1]
    aliascol1 = getColAlias(db, table1, col1)
    aliascol2 = getColAlias(db, table1, col2)

    decision = random.choice([(w_all, '*'), (aliascol1, col1)])  # decide to select all or select just the 1 column
    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    nlq = f"{q1} {decision[0]} {w_from} {gt_alias_table} {w_orderby} {aliascol2} {w_asc}"
    sql = f"SELECT {decision[1]} FROM {gt_table} ORDER BY {col2} ASC;"
    print("*****"*10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****"*10)
    return nlq, sql

def orderby3(db, table=None, tableLUT=None): #with descending conditioned on NUM type column
    table1, columns = get_table_k_columns(db, k=2, type_constraint="number", table=table)
    col1 = columns[0]
    col2 = columns[1]
    aliascol1 = getColAlias(db, table1, col1)
    aliascol2 = getColAlias(db, table1, col2)

    decision = random.choice([(w_all, '*'), (aliascol1, col1)])  # decide to select all or select just the 1 column

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    nlq = f"{q1} {decision[0]} {w_from} {gt_alias_table} {w_orderby} {aliascol2} {w_desc}"
    sql = f"SELECT {decision[1]} FROM {gt_table} ORDER BY {col2} DESC;"
    print("*****"*10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****"*10)
    return nlq, sql

#UPDATED: now needs to have 1 type number and 1 type text
def groupby1(db, table=None, tableLUT=None): #two columns, group by MIN of 2nd

    table1, col1 = get_table_k_columns(db, k=1, type_constraint="text", table=table)

    table2, col2 = get_table_k_columns(db, k=1, type_constraint="number", table=table1)

    aliascol1 = getColAlias(db, table1, col1)
    aliascol2 = getColAlias(db, table1, col2)


    degree = random.choice(["min", "max"])

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    nlq = f"{q1} {aliascol1} {w_and} {degree}imum {aliascol2} {w_from} {gt_alias_table} {w_groupby} {aliascol1} "
    sql = f"SELECT {col1}, {degree.upper()}({col2}) FROM {gt_table} GROUP BY {col1};"
    print("*****"*10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****"*10)
    return nlq, sql

def having1(db, table=None, tableLUT=None):
    table1, col1 = get_table_k_columns(db, k=1, type_constraint="text", table=table)
    table2, col2 = get_table_k_columns(db, k=1, type_constraint="number", table=table1)

    aliascol1 = getColAlias(db, table1, col1)
    aliascol2 = getColAlias(db, table1, col2)

    val2 = get_column_value(db, table1, col2)

    degree = random.choice(["min", "max"])

    operator = "equal to"
    symbol = "="

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1
    if val2 is None:
        return "BADVAL", "BADVAL"

    try:
        float(val2)
    except Exception as e:
        return "BADVAL", "BADVAL"

    nlq = f"{q1} {aliascol1} {w_from} {gt_alias_table} {w_groupby} {aliascol1} {w_having} {degree}imum {aliascol2} {operator} {val2}"
    sql = f"SELECT {col1} FROM {gt_table} GROUP BY {col1} HAVING {degree.upper()}({col2}) {symbol} {val2};"

    print("*****"*10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****"*10)
    return nlq, sql

######################## SQL FUNCTIONS ########################

def max1(db, table=None, tableLUT=None):
    table1, col1 = get_table_k_columns(db, k=1, type_constraint="number", table=table)
    aliascol1 = getColAlias(db, table1, col1)

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    nlq = f"{q1} {w_max} {aliascol1} {w_from} {gt_alias_table}"
    sql = f"SELECT MAX({col1}) FROM {gt_table} ;"
    print("*****" * 10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****" * 10)
    return nlq, sql

def min1(db, table=None, tableLUT=None):
    table1, col1 = get_table_k_columns(db, k=1, type_constraint="number", table=table)
    alias_col1 = getColAlias(db, table, col1)

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    nlq = f"{q1} {w_min} {alias_col1} {w_from} {gt_alias_table}"
    sql = f"SELECT MIN({col1}) FROM {gt_table} ;"
    print("*****" * 10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****" * 10)
    return nlq, sql

def sum1(db, table=None, tableLUT=None):
    table1, col1 = get_table_k_columns(db, k=1, type_constraint="number", table=table)
    aliascol1 = getColAlias(db, table1, col1)

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    nlq = f"{q1} {w_sum} {aliascol1} {w_from} {gt_alias_table}"
    sql = f"SELECT SUM({col1}) FROM {gt_table} ;"
    print("*****" * 10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****" * 10)
    return nlq, sql

def count1(db, table=None, tableLUT=None):
    table1, col1 = get_table_k_columns(db, k=1, table=table)
    alias_col1 = getColAlias(db, table, col1)

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    nlq = f"{q1} {w_count} {alias_col1} {w_from} {gt_alias_table}"
    sql = f"SELECT COUNT({col1}) FROM {gt_table} ;"
    print("*****" * 10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****" * 10)
    return nlq, sql

def average1(db, table=None, tableLUT=None):
    table1, col1 = get_table_k_columns(db, k=1, type_constraint="number", table=table)
    aliascol1 = getColAlias(db, table1, col1)

    if tableLUT is not None:  # if there is a tableLUT, update table1 to be the gt name
        lookup = tableLUT[table1]
        gt_alias_table = lookup[0]
        gt_table = lookup[1]
    else:
        gt_alias_table = table1
        gt_table = table1

    nlq = f"{q1} {w_avg} {aliascol1} {w_from} {gt_alias_table}"
    sql = f"SELECT AVG({col1}) FROM {gt_table} ;"
    print("*****" * 10)
    print(f"/ {nlq}")
    print(f'\\ {sql}')
    print("*****" * 10)
    return nlq, sql

################################################################
def make_all_examples(schema_graphs, compatibility_dict):

    list_of_data_point_dicts = [] #this is what we will output to json!

    dbruntimes = []
    t0 = time.time()

    pairs_per_rule = defaultdict(int)
    pairs_per_database = {}

    for db_name, db_id in list(schema_graphs.db_index.items()):
        if db_name.lower() != 'baseball_1': #because this DB sucks and breaks BERT
            print(f"db # {db_id}: {db_name}")
            db_start_time = time.time()
            working_db = schema_graphs[db_name]

            count_per_db = 0

            list_of_tables = list(working_db.table_names) #aliases!
            alias2TrueTable = aliasTable2TrueTableLUT(working_db) #{'shitty_alias': (true_alias, true_table)}
            gt_tables_db = [alias2TrueTable[t][1] for t in list_of_tables]


            #ignore any tables that are reserved sql words because this breaks the sql parser
            loop_tables = [t for t in list_of_tables if t.lower() not in sql_stopwords] #for itterating through

            for table in loop_tables:
                db_tables_dict = compatibility_dict[db_name]
                table_dict = db_tables_dict[table] #table aliases! not real table names
                numColumns = table_dict['numColumns'] #int
                numberColumnsTypeNumber = table_dict['numberColumnsTypeNumber'] #int
                numberColumnsTypeText = table_dict['numberColumnsTypeText'] #int
                picklistProblem = table_dict['picklistProblem'] #Bool


                if numColumns >= 2:
                    nlq, sql = selectk(working_db, k=random.choice([2,3,4,5,6,7,8,9,10]), table=table, tableLUT=alias2TrueTable) #every table needs at least 2 columns
                    list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                     "tables": gt_tables_db, "rule": "select"})

                    nlq, sql = orderby1(working_db, table=table, tableLUT=alias2TrueTable) #every table needs two columns
                    list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                     "tables": gt_tables_db, "rule": "orderby"})

                    count_per_db += 2
                    pairs_per_rule["selectk"] += 1
                    pairs_per_rule["orderby1"] += 1

                if numColumns >= 1:
                    nlq, sql = select1(working_db, table=table, tableLUT=alias2TrueTable) #every table needs at least 1 column
                    list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                     "tables": gt_tables_db, "rule": "select"})

                    nlq, sql = distinct1(working_db, table=table, tableLUT=alias2TrueTable) #every table needs at least 1 column
                    list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                     "tables": gt_tables_db, "rule": "distinct"})

                    # nlq, sql = from1(working_db, table=table, tableLUT=alias2TrueTable) #every table needs at least 1 column
                    # list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                    #                                  "tables": gt_tables_db, "rule": "from"})

                    nlq, sql = count1(working_db, table=table, tableLUT=alias2TrueTable) #needs 1 column
                    list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                     "tables": gt_tables_db, "rule": "count"})

                    count_per_db += 4
                    pairs_per_rule["select1"] += 1
                    pairs_per_rule["distinct1"] += 1
                    # pairs_per_rule["from1"] += 1
                    pairs_per_rule["count1"] += 1

                if numberColumnsTypeNumber >=2:
                    nlq, sql = orderby2(working_db, table=table, tableLUT=alias2TrueTable) #every table needs two clumns that are NUMBER type
                    list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                     "tables": gt_tables_db, "rule": "orderby"})

                    nlq, sql = orderby3(working_db, table=table, tableLUT=alias2TrueTable) #every table needs two columns that are NUMBER type
                    list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                     "tables": gt_tables_db, "rule": "orderby"})


                    count_per_db += 3
                    pairs_per_rule["orderby2"] += 1
                    pairs_per_rule["orderby3"] += 1

                if numberColumnsTypeNumber >=1:
                    nlq, sql = max1(working_db, table=table, tableLUT=alias2TrueTable) #needs 1 column number type
                    list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                     "tables": gt_tables_db, "rule": "max"})

                    nlq, sql = min1(working_db, table=table, tableLUT=alias2TrueTable) #needs 1 column number type
                    list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                     "tables": gt_tables_db, "rule": "min"})

                    nlq, sql = sum1(working_db, table=table, tableLUT=alias2TrueTable) #needs 1 column number type
                    list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                     "tables": gt_tables_db, "rule": "sum"})

                    nlq, sql = average1(working_db, table=table, tableLUT=alias2TrueTable) #needs 1 column number type
                    list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                     "tables": gt_tables_db, "rule": "average"})

                    count_per_db += 4
                    pairs_per_rule["max1"] += 1
                    pairs_per_rule["min1"] += 1
                    pairs_per_rule["sum1"] += 1
                    pairs_per_rule["average1"] += 1

                if numColumns >=2 and not picklistProblem:
                    print(f"DB-TABLE: {db_name}-{table}")
                    print(f"* num columns is {numColumns} and picklistproblem should be false --> {picklistProblem}")
                    nlq, sql = where1(working_db, table=table, tableLUT=alias2TrueTable)  # needs two columns. All columns need picklists for picking var.
                    if nlq != "BADVAL":
                        list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                         "tables": gt_tables_db, "rule": "where"})
                        pairs_per_rule["where1"] += 1

                    nlq, sql = where2(working_db, table=table, tableLUT=alias2TrueTable)
                    if nlq != "BADVAL":
                        list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                         "tables": gt_tables_db, "rule": "where"})
                        pairs_per_rule["where2"] += 1

                    nlq, sql = whereNotEquals(working_db, table=table, tableLUT=alias2TrueTable)
                    if nlq != "BADVAL":
                        list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                         "tables": gt_tables_db, "rule": "notequal"})
                        pairs_per_rule["notequals"] += 1


                    list_of_pairs, count_per_method = whereNested1(working_db, k=random.choice([3,4,5,6]), table=table, tableLUT=alias2TrueTable)
                    for pair in list_of_pairs:
                        nlq = pair[0]
                        sql = pair[1]
                        rulename = pair[2]
                        pairs_per_rule[rulename] += 1
                        list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                         "tables": gt_tables_db, "rule": rulename})

                    count_per_db += 2 + count_per_method
                    pairs_per_rule["whereNested1"] += count_per_method

                if numberColumnsTypeNumber >=1 and numberColumnsTypeText >=1:
                    nlq, sql = groupby1(working_db, table=table,
                                        tableLUT=alias2TrueTable)  # every table needs two columns that are NUMBER type
                    list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                     "tables": gt_tables_db, "rule": "groupby"})

                    pairs_per_rule["groupby1"] += 1

                if numberColumnsTypeNumber >= 1 and numberColumnsTypeText >= 1 and not picklistProblem:
                    nlq, sql = having1(working_db, table=table,
                                       tableLUT=alias2TrueTable)  # every table needs two columns that are NUMBER type
                    if nlq != "BADVAL":
                        list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                         "tables": gt_tables_db, "rule": "having"})
                        count_per_db += 1
                        pairs_per_rule["having1"] += 1


                if numberColumnsTypeNumber >=2 and not picklistProblem:
                    nlq, sql = whereMaths(working_db, table=table, tableLUT=alias2TrueTable)
                    if nlq != "BADVAL":
                        list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                         "tables": gt_tables_db, "rule": "maths"})
                        pairs_per_rule["maths"] += 1

                if numberColumnsTypeText >=2 and not picklistProblem:
                    list_of_pairs, count_per_method = whereNested2(working_db, k=2, table=table, tableLUT=alias2TrueTable) #this only contains like1
                    for pair in list_of_pairs:
                        nlq = pair[0]
                        sql = pair[1]
                        rulename = pair[2]
                        pairs_per_rule[rulename] += 1
                        list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                         "tables": gt_tables_db, "rule": rulename})

                        count_per_db +=1
                        pairs_per_rule["whereNested2"] += 1

                if numberColumnsTypeNumber >=3 and not picklistProblem:
                    try:
                        nlq, sql = whereNested3(working_db, k=random.choice([3,4,5,6]), table=table, tableLUT=alias2TrueTable)
                        if nlq != "BADVAL":
                            list_of_data_point_dicts.append({"db_id": db_name, "query": sql, "question": nlq,
                                                             "tables": gt_tables_db, "rule": "between"})

                            count_per_db += 1
                            pairs_per_rule["between1"] += 1
                            pairs_per_rule["whereNested3"] += 1

                    except Exception as e_oneValue:
                        #the selected num column probably only had one entry in it to fail whereNested3
                        pass


        pairs_per_database[db_name] = count_per_db

        dbruntimes.append(time.time() - db_start_time)

    print(f"* entire code executed in {time.time() - t0} ! ")
    print(f"* average db runtime for generating samples for all tables is: {sum(dbruntimes)/len(dbruntimes)}")
    for t, db_info in sorted(zip(dbruntimes, list(schema_graphs.db_index.items()))):
        db_tables_dict = compatibility_dict[db_info[0]]
        print(f"* {db_info[0].upper()} ({len(db_tables_dict)} tables):: {t} seconds - ")
    print("******** PAIRS PER DATABASE: ")
    print(pairs_per_database)
    print("******** PAIRS PER RULE: ")
    print(pairs_per_rule)
    print("******** OUTPUTTING TO JSONS: ")
    # #sort by RULE
    rule_examples_dict = defaultdict(list) #dictionary {rule: [dict, dict, dict], ...}
    for d in list_of_data_point_dicts:
        rule = d["rule"]
        rule_examples_dict[rule].append(d)

    for r, list_of_d in rule_examples_dict.items():
        new_dir = "data/scfg"
        Path(new_dir).mkdir(parents=True, exist_ok=True)
        new_file = f"scfg_data_{r}.json"
        print(f"NEW FILE: {new_file}")
        with open(os.path.join(new_dir, new_file), 'w') as outfile:
            json.dump(list_of_d, outfile, indent=4)

    print("* done :)")


def main(args):
    schema_graphs = load_schema_graphs_spider(args["DATAPATH"], args["DATASET"],
                                              db_dir="data/spider/database",
                                              augment_with_wikisql=False)

    if args["MODE"] == "all":
        compatibility_dict = precompute_compatibility(schema_graphs)
        #print(compatibility_dict)
        make_all_examples(schema_graphs, compatibility_dict)

    elif args["MODE"] == "one":
        #No checks for if DB/table meets criterion in this mode!
        print(f"* [WARNING !!!!!] 'one' mode does not guarantee the DB/table meets the criterion for these tests! Code may likely crash")

        # print(schema_graphs.db_index.keys()) #can get the names of the db from here with `.db_index`

        if args["DB"] == "random":
            db_name, db_id = random.choice(list(schema_graphs.db_index.items())) #get a random db and its id
            print(f"* randomly chosen DB: {db_name}")
        else:
            db_name = args["DB"]
        try:
            working_db = schema_graphs[db_name]
            print(f"* num tables: {working_db.num_tables}")
        except Exception as e:
            print(f"* Sorry, you've chosen a DB that apparently doesn't exist: {args['DB']} ... Check your spelling and try again. Exit code: 123")
            exit(123)

        select1(working_db)
        # selectk(working_db, k=4)
        # distinct1(working_db)
        # from1(working_db)
        # where1(working_db)
        # where2(working_db)
        # whereNested1(working_db, k=3)
        # whereNested2(working_db, k=3)
        # whereNested3(working_db, k=3)
        # orderby1(working_db)
        # orderby2(working_db)
        # orderby3(working_db)
        # groupby1(working_db)
        # having1(working_db)
        # max1(working_db)
        # min1(working_db)
        # sum1(working_db)
        # count1(working_db)
        # average1(working_db)



if __name__ == "__main__":
    args = docopt(__doc__)
    log.debug(args)
    log_level = logging.DEBUG if args["--debug"] else logging.INFO
    logzero.loglevel(log_level)
    logzero.formatter(logzero.LogFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
    main(args)
    print("* end")