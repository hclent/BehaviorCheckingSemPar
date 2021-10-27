"""
Data processing utilities.
"""

import random


# Dataset
WIKISQL = 0
SPIDER = 1
OTHERS = 2


class Example(object):
    """
    An example object stores a natural language question and the corresponding program
    translations in the following format:
        1. raw data
        2. vectorized program syntax components.

    A question may correspond to multiple correct programs.
    """

    def __init__(self, dataset_id):
        self.dataset_id = dataset_id

        self.text = None

        self.text_tokens = None
        self.input_tokens = None

        self.text_ptr_values = None
        self.input_ptr_values = None

        self.gt_program_list = []
        self.program_list = []

        self.program_tokens_list = []
        self.program_ast_list = []
        self.program_denormalized_ast_list = []

        self.program_tokens_list_ = []      # official program tokenization (if applicable)
        self.program_ast_list_ = []         # official program AST (if applicable)

        self.text_ids = None
        self.text_ptr_input_ids = None
        self.text_ptr_value_ids = None
        self.ptr_input_ids = None
        self.ptr_value_ids = None

        self.program_input_ids_list = []
        self.program_text_ptr_value_ids_list = []
        self.program_text_span_ptr_ids_list = []

        self.variables = {}

        # by default, use the first program in the ground truth list
        self.program_id = 0

    def add_program(self, program, program_ast=None, program_tokens=None):
        self.program_list.append(program)
        if program_ast is not None:
            self.program_ast_list.append(program_ast)
        if program_tokens is not None:
            self.program_tokens_list.append(program_tokens)

    def add_program_official(self, program, program_ast=None, program_tokens=None):
        self.program_list.append(program)
        if program_ast is not None:
            self.program_ast_list_.append(program_ast)
        if program_tokens is not None:
            self.program_tokens_list_.append(program_tokens)

    def set_program_id(self):
        num_programs = self.num_programs
        if num_programs > 1:
            self.program_id = random.randint(0, num_programs - 1)

    def pretty_print(self, example_id=None, schema=None, de_vectorize_ptr=None, rev_vocab=None, post_process=None,
                     use_table_aware_te=True):
        if example_id:
            print('Example {}'.format(example_id))
        if schema:
            schema.pretty_print()
        print('NL: {}'.format(self.text.encode('utf-8')))
        print('NL tokens: {}'.format([t.encode('utf-8') for t in self.text_tokens]))
        print('NL tokens (original): {}'.format([t.encode('utf-8') for t in self.text_ptr_values]))
        for i, program in enumerate(self.program_list):
            print('Target {}: {}'.format(i, program.encode('utf-8')))
            if i < len(self.program_tokens_list):
                program_tokens = self.program_tokens_list[i]
                print('Target tokens: {}'.format([t.encode('utf-8') for t in program_tokens]))
            if i < len(self.program_text_ptr_value_ids_list):
                input_tokens = self.input_ptr_values if use_table_aware_te else self.text_ptr_values
                program_tokens = de_vectorize_ptr(
                    self.program_text_ptr_value_ids_list[i], rev_vocab, input_tokens, post_process, return_tokens=True)
                print('Target T-P tokens: {}'.format(program_tokens))
        print()

    def run_unit_tests(self):
        # assert(len(self.text_ptr_values) == len(self.text_ptr_input_ids))
        assert(len(self.text_ptr_values) == len(self.text_ptr_value_ids))
        assert(len(self.input_tokens) == len(self.ptr_input_ids))
        assert(len(self.input_ptr_values) == len(self.ptr_input_ids))
        assert(not self.program_tokens_list or len(self.program_tokens_list) == self.num_programs)
        assert(not self.program_ast_list or len(self.program_ast_list) == self.num_programs)
        assert(not self.program_tokens_list_ or len(self.program_tokens_list_) == self.num_programs)
        assert(not self.program_ast_list_ or len(self.program_ast_list_) == self.num_programs)
        assert(not self.program_input_ids_list or len(self.program_input_ids_list) == self.num_programs)
        assert(not self.program_text_ptr_value_ids_list or len(self.program_text_ptr_value_ids_list) == self.num_programs)
        assert (not self.program_text_span_ptr_ids_list or len(self.program_text_span_ptr_ids_list) == self.num_programs)

    @property
    def num_programs(self):
        return len(self.program_list)

    @property
    def program_len(self):
        return len(self.program_input_ids)

    @property
    def program(self):
        return self.program_list[self.program_id]

    @property
    def program_tokens(self):
        return self.program_tokens_list[self.program_id]

    @property
    def program_input_ids(self):
        return self.program_input_ids_list[self.program_id]

    @property
    def program_ast(self):
        return self.program_ast_list[self.program_id]

    @property
    def program_text_ptr_value_ids(self):
        return self.program_text_ptr_value_ids_list[self.program_id]

    @property
    def program_text_span_ptr_ids(self):
        return self.program_text_span_ptr_ids_list[self.program_id]

    @property
    def num_text_tokens(self):
        return len(self.text_ids)

    @property
    def num_input_tokens(self):
        return len(self.input_tokens)

    @property
    def num_program_tokens(self):
        return len(self.program_input_ids)


class TableSemanticParsingExample(Example):

    def __init__(self, dataset_id, db_name, db_id):
        super().__init__(dataset_id)
        self.db_name = db_name
        self.db_id = db_id

        self.schema_M = None
        self.M = None

        self.gt_tables_list = []
        self.gt_table_names_list = []
        self.gt_fields_list = []

        self.transformer_output_value_masks = None

        self.pred_tables = None
        self.table_ids_list = []

    def add_gt_tables(self, gt_tables, gt_table_names):
        self.gt_tables_list.append(gt_tables)
        self.gt_table_names_list.append(gt_table_names)

    def pretty_print(self, example_id=None, schema=None, de_vectorize_ptr=None, rev_vocab=None, post_process=None,
                     use_table_aware_te=True):
        if example_id:
            print('Example {}'.format(example_id))
        if schema:
            schema.pretty_print()
        print('NL: {}'.format(self.text.encode('utf-8')))
        print('NL tokens: {}'.format([t.encode('utf-8') for t in self.text_tokens]))
        print('NL tokens (original): {}'.format([t.encode('utf-8') for t in self.text_ptr_values]))
        for i, program in enumerate(self.program_list):
            print('Target {}: {}'.format(i, program.encode('utf-8')))
            if i < len(self.program_tokens_list):
                program_tokens = self.program_tokens_list[i]
                print('Target tokens: {}'.format([t.encode('utf-8') for t in program_tokens]))
            if i < len(self.program_text_ptr_value_ids_list):
                input_tokens = self.input_ptr_values if use_table_aware_te else self.text_ptr_values
                program_tokens = de_vectorize_ptr(
                    self.program_text_ptr_value_ids_list[i], rev_vocab, input_tokens, post_process, return_tokens=True)
                print('Target T-P tokens: {}'.format(program_tokens))
        print()

    @property
    def gt_tables(self):
        return self.gt_tables_list[self.program_id]

    @property
    def gt_table_names(self):
        return self.gt_table_names_list[self.program_id]

    @property
    def gt_fields(self):
        return self.gt_fields_list[self.program_id]

    @property
    def table_ids(self):
        return self.table_ids_list[self.program_id]


class Text2SQLExample(TableSemanticParsingExample):

    def __init__(self, dataset_id, db_name, db_id):
        super().__init__(dataset_id, db_name, db_id)
        self.primary_key_ids = None
        self.foreign_key_ids = None
        self.field_type_ids = None
        self.task_masks = None

        self.program_singleton_field_tokens_list = []
        self.program_singleton_field_token_types_list = []
        self.program_singleton_field_input_ids_list = []
        self.program_text_and_field_ptr_value_ids_list = []

        self.leaf_condition_vals_list = []
        self.leaf_condition_val_ids_list = []
        self.leaf_condition_val_ptr_ids_list = []

        self.select_clause_vec_list = []
        self.where_clause_vec_list = []
        self.group_by_clause_vec_list = []
        self.order_by_clause_vec_list = []
        self.leaf_condition_op_ids_list = []

    def run_unit_tests(self):
        super().run_unit_tests()
        assert(not self.program_singleton_field_input_ids_list or
               len(self.program_singleton_field_input_ids_list) == self.num_programs)
        assert(not self.program_text_and_field_ptr_value_ids_list or
               len(self.program_text_and_field_ptr_value_ids_list) == self.num_programs)

    def pretty_print(self, example_id=None, schema=None, de_vectorize_ptr=None, de_vectorize_field_ptr=None,
                     rev_vocab=None, post_process=None, use_table_aware_te=True):
        if example_id:
            print('Example {}'.format(example_id))
        if schema:
            schema.pretty_print()
        print('NL: {}'.format(self.text.encode('utf-8')))
        print('NL tokens: {}'.format([t.encode('utf-8') for t in self.text_tokens]))
        print('NL tokens (original): {}'.format([t.encode('utf-8') for t in self.text_ptr_values]))
        for i, program in enumerate(self.program_list):
            print('Target {}: {}'.format(i, program.encode('utf-8')))
            if i < len(self.program_tokens_list):
                program_tokens = self.program_tokens_list[i]
                print('Target tokens: {}'.format([t.encode('utf-8') for t in program_tokens]))
            if i < len(self.program_text_ptr_value_ids_list):
                input_tokens = self.input_ptr_values if use_table_aware_te else self.text_ptr_values
                program_tokens = de_vectorize_ptr(
                    self.program_text_ptr_value_ids_list[i], rev_vocab, input_tokens, post_process, return_tokens=True)
                print('Target T-P tokens: {}'.format(program_tokens))
            if schema and i < len(self.program_text_and_field_ptr_value_ids_list):
                program_tokens = de_vectorize_field_ptr(
                    self.program_text_and_field_ptr_value_ids_list[i], rev_vocab, self.text_ptr_values, schema=schema,
                    post_process=post_process, return_tokens=False)
                print('Target TF-P tokens: {}'.format(program_tokens))
        print()

    @property
    def program_singleton_field_tokens(self):
        return self.program_singleton_field_tokens_list[self.program_id]

    @property
    def program_singleton_field_token_types(self):
        return self.program_singleton_field_token_types_list[self.program_id]

    @property
    def program_singleton_field_input_ids(self):
        return self.program_singleton_field_input_ids_list[self.program_id]

    @property
    def program_text_and_field_ptr_value_ids(self):
        return self.program_text_and_field_ptr_value_ids_list[self.program_id]

    @property
    def select_clause_vec(self):
        return self.select_clause_vec_list[self.program_id]

    @property
    def where_clause_vec(self):
        return self.where_clause_vec_list[self.program_id]

    @property
    def group_by_clause_vec(self):
        return self.group_by_clause_vec_list[self.program_id]

    @property
    def order_by_clause_vec(self):
        return self.order_by_clause_vec_list[self.program_id]

    @property
    def leaf_co_ids(self):
        return self.leaf_condition_op_ids_list[self.program_id]

    @property
    def leaf_cv_ids(self):
        return self.leaf_condition_val_ids_list[self.program_id]

    @property
    def leaf_cv_ptr_ids(self):
        return self.leaf_condition_val_ptr_ids_list[self.program_id]

    @property
    def leaf_cv_vals(self):
        return self.leaf_condition_vals_list[self.program_id]


class AugmentedText2SQLExample(object):
    """
    An text-to-SQL example introduced via data augmentation that contains a pointer to the original example and data
    entries for training.
    """
    def __init__(self, example, db_name, db_id):
        self.example = example
        self.db_name = db_name
        self.db_id = db_id

        self.schema_M = None
        self.M = None

        self.gt_tables_list = []
        self.gt_fields_list = []

        self.input_tokens = None
        self.input_ptr_values = None
        self.ptr_input_ids = None
        self.ptr_value_ids = None

        self.transformer_output_value_masks = None

        self.primary_key_ids = None
        self.foreign_key_ids = None
        self.field_type_ids = None
        self.task_masks = None

        self.program_singleton_field_input_ids_list = []
        self.program_text_and_field_ptr_value_ids_list = []

        self.pred_tables = None
        self.table_ids_list = []

        self.leaf_condition_val_ids_list = []
        self.leaf_condition_val_ptr_ids_list = []

        self.select_clause_vec_list = []
        self.where_clause_vec_list = []
        self.group_by_clause_vec_list = []
        self.order_by_clause_vec_list = []
        self.leaf_condition_op_ids_list = []

        # by default, use the first program in the ground truth list
        self.program_id = 0

    def run_unit_tests(self):
        assert (not self.program_singleton_field_input_ids_list or
                len(self.program_singleton_field_input_ids_list) == self.example.num_programs)
        assert(not self.program_text_and_field_ptr_value_ids_list or
               len(self.program_text_and_field_ptr_value_ids_list) == self.example.num_programs)

    def pretty_print(self, *args, **kwargs):
        self.example.pretty_print(*args, **kwargs)


    @property
    def text(self):
        return self.example.text

    @property
    def text_ids(self):
        return self.example.text_ids

    @property
    def text_tokens(self):
        return self.example.text_tokens

    @property
    def text_ptr_values(self):
        return self.example.text_ptr_values

    @property
    def text_ptr_value_ids(self):
        return self.example.text_ptr_value_ids

    @property
    def gt_tables(self):
        return self.gt_tables_list[self.program_id]

    @property
    def gt_table_names(self):
        return self.example.gt_table_names_list[self.program_id]

    @property
    def gt_table_names_list(self):
        return self.example.gt_table_names_list

    @property
    def table_ids(self):
        return self.table_ids_list[self.program_id]

    @property
    def program_input_ids(self):
        return self.example.program_input_ids_list[self.program_id]

    @property
    def program_singleton_field_tokens(self):
        return self.example.program_singleton_field_tokens_list[self.program_id]

    @property
    def program_singleton_field_token_types(self):
        return self.example.program_singleton_field_token_types_list[self.program_id]

    @property
    def program_singleton_field_input_ids(self):
        return self.program_singleton_field_input_ids_list[self.program_id]

    @property
    def program_text_and_field_ptr_value_ids(self):
        return self.program_text_and_field_ptr_value_ids_list[self.program_id]

    @property
    def leaf_cv_ids(self):
        return self.leaf_condition_val_ids_list[self.program_id]

    @property
    def leaf_cv_ptr_ids(self):
        return self.leaf_condition_val_ptr_ids_list[self.program_id]

    @property
    def leaf_cv_vals(self):
        return self.example.leaf_condition_vals_list[self.program_id]

    @property
    def num_text_tokens(self):
        return self.example.num_text_tokens

    @property
    def num_input_tokens(self):
        return len(self.input_tokens)

    @property
    def num_program_tokens(self):
        return self.example.num_program_tokens



