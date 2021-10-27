# BehaviorCheckingSemPar

Evaluating Cross-Database Semantic Parsers Using Canonical Utterances


## Overview
Check back soon for Eval4nlp!!

## Quick Start

If you want our **test data from the paper**, this is located the under `eval4nlp/scfg`.

If you would like to create your own test data from scratch, or integrate your own SCFG rules, you can follow the instructions below! 

### Install Dependencies

First, set up a basic environment.
```
git clone https://github.com/hclent/BehaviorCheckingSemPar.git
cd BehaviorCheckingSemPar

conda create -n sempar python=3.6
conda activate sempar
pip install -r requirements.txt
```

### Prepare Spider Data
Unzip the `databases` in `data/spider`, before you get started. The data in this repo is **NOT out-of-the-box Spider**! 

Our version of Spider follows data pre-processing conventions introduced by [TabularSemanticParsing](https://github.com/salesforce/TabularSemanticParsing).
If you'd like to learn more, we recommend downloading this code and following the instructions for "Process Data".



### Generate Test Data

To generate test data, simply run:

`python grammar.py spider data/spider all random`

and then conver these outputs to match the complete Spider format:

```
python format_grammar.py
```


Our `grammar.py` uses the spider databases (`data/spider`) to generate SCFG test pairs for all categories of SQL elements, listed above.
Columns and values in resulting queries are chosen randomly (random seed 27).
This code is structured such that, for each database, for each table, all applicable SCFG rules will run. 
(E.g., if a given table only has `<number>` variables, the code will not return pairs for the `LIKE` operator).


### How can I add new rules? 

First, lets look at an example SCFG rule:
```
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
```

Each new SCFG rule should, at minimum, take the parameters `db`, `table=None`, `tableLUT=None`. Extra parameters can also be specified, such as 
`k=3` for number of columns, and so on. 
The most important part of making a new rule is to return `nlq, sql` pair, where the `nlq` uses elements from the global SCFG variables combined with alias names for tables and columns,
 and the `sql` uses the original table and column names, joined by SQL syntax.

A good way to sanity check your new rules, is to plug them into the very bottom of the `grammar.py`, and run 
`python grammar.py spider data/spider one random`. This picks a random dataset, to test your one rule on.
 You may also substitute a database name for `random` to develop with a certain databse in mind.
 
 Once you are happy with your rule, you can add it to the pipeline for executing `all` SCFG rules. A rule's place in the pipeline 
 is based on what table, column, and variable restrictions exist for a given SQL element. Start at L#1311, and follow along to see where you should plug in your rule. 
 

###Unit testing on downstream models?
Because this test suite outputs data in the Spider format, you will easily be able to "plug-and-play" this test data into other models already supporting Spider.

## Citation
Coming soon :-)
