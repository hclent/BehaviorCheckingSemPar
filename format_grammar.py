import os
from pathlib import Path
import json
from collections import defaultdict
from sqlalchemy import create_engine, MetaData, Table
from nltk import word_tokenize
import process_sql as ps
"""
For taking the output of grammar.py (scfg json files) and adding the other Spider fields to them
"""


def load_scfg_json(full_path):
    with open(full_path) as inj:
        entries = json.loads(inj.read())
    return entries

def getTrueSchemaAsDict(db):
    true_schema_graph = defaultdict(list)
    old_db_dir = "data/spider/database"
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


def get_tokens_for_entry(single_entry):
    query = single_entry["query"]
    #print(f"QUERY: {query}  (type({type(query)}))")
    question = single_entry["question"]
    db = single_entry["db_id"]

    question_toks = word_tokenize(question)
    #print(f"QUESTION TOKS: {question_toks}")

    query_toks = []
    query_toks_no_value = []

    toks = query.split(" ")
    for t in toks:
        if t.lower().startswith(("min", "max", "sum", "avg", "count", "distinct")):
            key_word = t[:t.find("(")]
            sub_token = t[t.find("(")+1:t.find(")")]
            query_toks.append(key_word)
            query_toks.append("(")
            query_toks.append(sub_token)
            query_toks.append(")")
        elif t.lower().startswith(("(select")):
            query_toks.append("(")
            query_toks.append("SELECT")
        elif t == ";":
            pass
        elif t.endswith(";"):
            query_toks.append(t[:-1])
        elif t.endswith(","):
            query_toks.append(t[:-1])
            query_toks.append(",")
        else:
            query_toks.append(t)

    for t in query_toks:
        if t.isnumeric():
            query_toks_no_value.append("value")
        elif "." in t:
            try:
                float(t)
                query_toks_no_value.append("value")
            except Exception as e:
                pass
        elif "'" in t:
            query_toks_no_value.append("value")
        else:
            query_toks_no_value.append(t.lower())

    return question_toks, query_toks, query_toks_no_value

def add_sql_dict_to_entry(single_entry):
    query = single_entry["query"]
    db = single_entry["db_id"]
    relevant_schema = getTrueSchemaAsDict(db)
    typed_schema = ps.Schema(relevant_schema)
    out_sql = ps.get_sql(typed_schema, query)
    return out_sql

def entry_to_schema_lookup():
    json_path = "data/spider"
    original_tables_json = os.path.join(json_path, "tables.json")
    with open(original_tables_json) as inj:
        original_tables = json.loads(inj.read())
    original_tables_as_dict = {}  # LUT db --> table.json entry
    for t_j in original_tables:
        original_tables_as_dict[t_j["db_id"]] = t_j
    return original_tables_as_dict


def make_new_file(base_path, f):
    full_path = os.path.join(base_path, f)
    old_entries_list = load_scfg_json(full_path)
    print(f"* {f}: {len(old_entries_list)}")

    new_entry_list = []

    for entry in old_entries_list:
        new_entry = {}
        question_toks, query_toks, query_toks_no_value = get_tokens_for_entry(entry)
        out_sql = add_sql_dict_to_entry(entry)
        for key, items in entry.items():
            new_entry[key] = items
        new_entry["question_toks"] = question_toks
        new_entry["query_toks"] = query_toks
        new_entry["query_toks_no_value"] = query_toks_no_value
        new_entry["sql"] = out_sql
        new_entry_list.append(new_entry)

    out_path = "data/scfglong"
    Path(out_path).mkdir(parents=True, exist_ok=True)
    new_file = os.path.join(out_path, f)
    with open(new_file, 'w') as outfile:
        json.dump(new_entry_list, outfile, indent=4)
    print(f"**** OUT: {new_file}")



def main():
    base_path = "data/scfg"
    f_list = os.listdir(base_path)
    print(f_list)
    for f in f_list:
        try:
            make_new_file(base_path, f)
        except Exception as e:
            print(f"{f}: {e}")


main()
print(f"* done")