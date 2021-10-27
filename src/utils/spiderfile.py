import re

sql_stopwords = ['and', 'as', 'asc', 'between', 'case', 'collate_nocase', 'cross_join', 'desc', 'else', 'end',
                 'from',
                 'full_join', 'full_outer_join', 'group_by', 'having', 'in', 'inner_join', 'is', 'is_not', 'join',
                 'left_join', 'left_outer_join', 'like', 'limit', 'none', 'not_between', 'not_in', 'not_like',
                 'offset',
                 'on', 'or', 'order_by', 'reserved', 'right_join', 'right_outer_join', 'select', 'then', 'union',
                 'union_all', 'except', 'intersect', 'using', 'when', 'where', 'binary_ops', 'unary_ops', 'with',
                 'durations', 'max', 'min', 'count', 'sum', 'avg', 'minimum', 'maximum', 'ascending', 'descending',
                 'average']


def newTableDotJsonItem(original_tables_entry, perturbed_key, key_change=False):
    """
    original_tables_entry: dict entry from tables.json
    perturbed_key: dict(original table/col --> perturbed table/col)
    key_change: Bool. Should be true when foreign/primary keys are fudged with
    """
    new_table_item = {}

    if key_change == False:  # if there's no fudjing with primary and foreign keys
        for k, objs in original_tables_entry.items():
            if k == "column_names":  # list of lists #TODO: give typos also to the "nice names". nice names needs to "match"
                new_table_item[k] = objs

            # NOTE: COLUMN_NAMES are lexical FEATURES of the ORIGINAL!!! Keep them // consistent.
            elif k == "column_names_original":  # list of lists
                print(f"umm hi?")
                print(f"OBJS: {objs}")
                print(type(objs))
                new_col_list = []
                for col_list in objs:
                    print(f"col list: {col_list} is type ({type(col_list)})")
                    num = col_list[0]
                    old_col_name = col_list[1]
                    if old_col_name in list(perturbed_key.keys()):
                        new_col_name = perturbed_key[old_col_name]
                        new_col_list.append([num, new_col_name])
                    else:
                        new_col_list.append([num, old_col_name])
                new_table_item[k] = new_col_list

            # NOTE: TABLE_NAMES are lexical FEATURES of the ORIGINAL!!! Keep them // consistent.
            elif k == "table_names":  # list #TODO: give typos also to the "nice names". nice names needs to "match"
                new_table_item[k] = objs
            elif k == "table_names_original":  # list
                new_tables = [perturbed_key[o] for o in objs]
                new_table_item[k] = new_tables
            else:
                new_table_item[k] = objs  # just keep whats there

    return new_table_item


def newTableDotJsonAddTable(original_tables_entry, add_tables_entry, add_schema_elem):
    """
    original_tables_entry: dict entry from tables.json
    add_tables_entry: dict entry from tables.json for the db of the table you are adding to the schema
    add_schema_elem: dict. {table: [col, col, col]}
    """
    new_table_item = {}

    new_table_names_original = original_tables_entry["table_names_original"]
    new_table_names = original_tables_entry["table_names"]
    new_column_names_original = original_tables_entry["column_names_original"]
    # [ [-1,"*"],[0,"Stadium_ID"], [0, blah], [], ... ]
    new_column_names = original_tables_entry["column_names"]
    new_column_types = original_tables_entry["column_types"]

    add_column_indexes = {}
    native_add_columns = []
    nice_add_columns = []
    for add_table, add_columns in add_schema_elem.items():
        new_table_names_original.append(add_table)
        add_table_index = add_tables_entry["table_names_original"].index(add_table)
        nice_table_name = add_tables_entry["table_names"][add_table_index]
        new_table_names.append(nice_table_name)

        for col_elem in add_tables_entry["column_names_original"]:
            if col_elem[0] == add_table_index:
                native_add_columns.append(col_elem)
        for col_elem in add_tables_entry["column_names"]:
            if col_elem[0] == add_table_index:
                nice_add_columns.append(col_elem)
        for col in add_columns:
            for col_elem in add_tables_entry["column_names_original"]:
                if col == col_elem[1]:
                    col_index = add_tables_entry["column_names_original"].index(col_elem)
                    add_column_indexes[col] = col_index

    # print(f"native add columns: {native_add_columns}")
    # print(f"nice add columns: {nice_add_columns}")
    # print(f"add_column_indexes: {add_column_indexes}")

    # now change the number in native_add_columns and nice_add_columns
    correct_new_table_index = len(original_tables_entry["table_names_original"]) - 1  # because there's a -1 entry
    corrected_native_add_columns = []
    corrected_nice_add_columns = []
    for col_elem in native_add_columns:
        e1 = correct_new_table_index
        e2 = col_elem[1]
        new_elem = [e1, e2]
        corrected_native_add_columns.append(new_elem)

    for col_elem in nice_add_columns:
        e1 = correct_new_table_index
        e2 = col_elem[1]
        new_elem = [e1, e2]
        corrected_nice_add_columns.append(new_elem)

    [new_column_names_original.append(corr) for corr in corrected_native_add_columns]
    [new_column_names.append(corr) for corr in corrected_nice_add_columns]

    # now get the right column types
    ordered_add_column_indexes = [add_column_indexes[ac[1]] for ac in native_add_columns]
    add_types = []
    for i in ordered_add_column_indexes:
        add_types.append(add_tables_entry["column_types"][i])
    [new_column_types.append(atype) for atype in add_types]

    new_table_item["table_names_original"] = new_table_names_original
    new_table_item["table_names"] = new_table_names
    new_table_item["column_names_original"] = new_column_names_original
    new_table_item["column_names"] = new_column_names
    new_table_item["column_types"] = new_column_types
    for key, items in original_tables_entry.items():
        if key in ["db_id", "foreign_keys", "primary_keys"]:
            new_table_item[key] = items

    return new_table_item


def newDevDotJsonItem(true_schema_graph, original_dev_entries, peturbed_key, key_change=False):
    """
    #TODO: table_names, query_toks, and query_toks_no_value also need to be changed to deal with the same case changing BS in dev.

    Input:
        true_schema_graph: dict{table: [col, col, col]} of the ORIGINAL Db, before perturbations
        original_dev_entries: list[{}, {}, {}] List of dicts, where each dict is a dev.json entry (see below)
        peturbed_key: {originalDBelement: perturbedElement}
        key_change: Bool. [default=False]. This is for if we have to do any shenanigans with primary/foreign keys...


    EXAMPLE DEV.JSON ENTRY
    {'db_id': 'concert_singer',
    'query': 'SELECT count(*) FROM singer',  <--------- this is not necessarily easy to change???
    'query_toks': ['SELECT', 'count', '(', '*', ')', 'FROM', 'singer'],
    'query_toks_no_value': ['select', 'count', '(', '*', ')', 'from', 'singer'],
    'question': 'How many singers do we have?',
    'question_toks': ['How', 'many', 'singers', 'do', 'we', 'have', '?'],
    'sql': {'except': None, 'from': {'conds': [], 'table_units': [['table_unit', 1]]},
        'groupBy': [], 'having': [], 'intersect': None, 'limit': None, 'orderBy': [],
        'select': [False, [[3, [0, [0, 0, False], None]]]], 'union': None, 'where': []},
        'tables': [1], 'table_names': ['singer']}

    Output:
        list_example_dicts: list[{}, {}, {}] A list of modified dev.json dicts, to be output
    """
    print(f"***** PERTURBED KEY: {peturbed_key}")
    # make a map of {original.lower(): original}, which can then be used for when the table/column has been lowercased
    lowerupperLUT = {}
    for elem in list(peturbed_key.keys()):
        lowerupperLUT[elem.lower()] = elem
        lowerupperLUT[elem.upper()] = elem

    list_example_dicts = []
    for entry in original_dev_entries:
        new_dev_example = {}
        #if key_change is False:
        for key, item in entry.items():
            if key == "query_toks" or key == "query_toks_no_value":
                new_query_toks = []
                for tok in item:

                    if tok.lower() in sql_stopwords:
                        new_query_toks.append(tok)  # just append the thingy and move on...

                    # otherwise if its just like... obvi a pertrubed thing, perturb it and move on
                    elif tok in list(peturbed_key.keys()):
                        new_query_toks.append(peturbed_key[tok])

                    elif re.match('(^T|t)\d\.', tok):  # its T1.something

                        base = tok[:3]
                        t_name = tok[3:]

                        # first try to see if the element as it is, is fine:
                        if t_name in list(peturbed_key.keys()):
                            new_t = base + peturbed_key[t_name]  # gotta add back on the T#. stuff
                            new_query_toks.append(new_t)  # we good and move on!

                        else:  # if not?? we gotta try cleaning it up first!

                            try:

                                # a thing can need to get stripped, and then still be a fucked case ...
                                # Now lets check if they fucked with the casing
                                if t_name == t_name.lower():  # update t_name to be the true name
                                    lookup_name = lowerupperLUT[t_name]

                                elif t_name == t_name.upper():  # if they lower- or upper- cased
                                    lookup_name = lowerupperLUT[t_name]

                                elif t_name.lower() in list(
                                        lowerupperLUT.keys()):  # to handle stupid things where the casing is totally wrong
                                    dummy_name = t_name.lower()
                                    lookup_name = lowerupperLUT[
                                        dummy_name]  # so now this maps to the original, which then we can peturb key to get the perturb!
                                else:
                                    lookup_name = t_name

                                new_t = base + peturbed_key[
                                    lookup_name]  # make this work with lowercased candidates
                                new_query_toks.append(new_t)
                            except Exception as e:
                                new_query_toks.append(tok)
                                print(f"-----------------")
                                print(f"PROBLEM CANDIDATE: {tok[3:]} --> lookup name: {lookup_name}")
                                print(f"perturbed key: {peturbed_key}")
                                print(f"----- {e} -------")


                    else: #see if they did any hurrendous casing changes to it

                        # elif see if its in the perturbed thing, but they messed up the casing
                        if tok == tok.lower() and tok in list(
                                lowerupperLUT.keys()):  # its lowercased and in the thing. look up and check if its a thingy
                            real_tok = lowerupperLUT[tok]
                            new_query_toks.append(peturbed_key[real_tok])

                        # elif see if its in the perturbed thing, but they messed up the casing
                        elif tok == tok.upper() and tok in list(
                                lowerupperLUT.keys()):  # its uppercased and in the thing
                            real_tok = lowerupperLUT[tok]
                            new_query_toks.append(peturbed_key[real_tok])

                        ##elif see if its in the perturbed thing, but they messed up the casing
                        elif tok.lower() in list(
                                lowerupperLUT.keys()):  # dealing with when the casing is just totally wrong, match the lowercase
                            real_tok = lowerupperLUT[tok.lower()]
                            new_query_toks.append(peturbed_key[real_tok])
                            """
                            perturbed_key = {'Hight_definition_TV': 'HightDefinitionTV', ...}

                            but the token is: hight_definition_TV

                            Hight_definition_TV --> hight_definition_tv --> HightDefinitionTV.
                            ^ Go through the lowercase to get to the proper one
                            """
                        else:  # just add it
                            new_query_toks.append(tok)


                assert len(new_query_toks) == len(item)
                new_dev_example[key] = new_query_toks
                if key == "query_toks":
                    new_dev_example["query"] = (' ').join(new_query_toks)

            elif key == "sql":
                new_sql_dict = {}
                for k, i in item.items():
                    if k == "table_names":
                        new_i = []
                        for tab in i:
                            """    
                            new_i = [peturbed_key[t] for t in i]
                            new_sql_dict[k] = new_i
                            """

                            # otherwise if its just like... obvi a pertrubed thing, perturb it and move on
                            if tab in list(peturbed_key.keys()):
                                new_i.append(peturbed_key[tab])

                            else:  # see if they did any hurrendous casing changes to it

                                # elif see if its in the perturbed thing, but they messed up the casing
                                if tab == tab.lower() and tab in list(
                                        lowerupperLUT.keys()):  # its lowercased and in the thing. look up and check if its a thingy
                                    real_tab = lowerupperLUT[tab]
                                    new_i.append(peturbed_key[real_tab])

                                # elif see if its in the perturbed thing, but they messed up the casing
                                elif tab == tab.upper() and tab in list(
                                        lowerupperLUT.keys()):  # its uppercased and in the thing
                                    real_tab = lowerupperLUT[tab]
                                    new_i.append(peturbed_key[real_tab])

                                ##elif see if its in the perturbed thing, but they messed up the casing
                                elif tab.lower() in list(
                                        lowerupperLUT.keys()):  # dealing with when the casing is just totally wrong, match the lowercase
                                    real_tab= lowerupperLUT[tab.lower()]
                                    new_i.append(peturbed_key[real_tab])
                                    """
                                    perturbed_key = {'Hight_definition_TV': 'HightDefinitionTV', ...}

                                    but the token is: hight_definition_TV

                                    Hight_definition_TV --> hight_definition_tv --> HightDefinitionTV.
                                    ^ Go through the lowercase to get to the proper one
                                    """
                                else:  # just add it
                                    new_i.append(tab)

                        assert len(new_i) == len(i)
                        new_sql_dict[k] = new_i
                    else:
                        new_sql_dict[k] = i
            else:
                if key != "query": #dont print the query key twice
                    new_dev_example[key] = item
        list_example_dicts.append(new_dev_example)
    print(f"--------------------------------------------------------------------")
    return list_example_dicts

