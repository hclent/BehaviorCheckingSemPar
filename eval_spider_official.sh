#!/usr/bin/env bash


DATA_DIR="data/spider/"
#GOLD_SQL_FILE="${DATA_DIR}dev_gold.sql"
PRED_SQL_FILE=$1
GOLD_SQL_FILE=$2
DB_DIR="${DATA_DIR}database/"
TABLE_FILE="${DATA_DIR}tables.json"
ETYPE="all"

cmd="python3 -m src.eval.spider.evaluate \
    --gold ${GOLD_SQL_FILE} \
    --pred ${PRED_SQL_FILE} \
    --db ${DB_DIR} \
    --table ${TABLE_FILE} \
    --etype ${ETYPE}"

echo "run ${cmd}"
${cmd}
