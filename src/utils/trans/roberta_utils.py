"""
Huggingface pretrained RoBERta model.
"""

from transformers import RobertaTokenizer


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
rt = tokenizer

pad_token = rt.pad_token
cls_token = rt.cls_token
sep_token = rt.sep_token
mask_token = rt.mask_token
unk_token = tokenizer.unk_token
pad_id = rt.convert_tokens_to_ids(pad_token)
cls_id = rt.convert_tokens_to_ids(cls_token)
sep_id = rt.convert_tokens_to_ids(sep_token)
mask_id = rt.convert_tokens_to_ids(mask_token)
unk_id = rt.convert_tokens_to_ids(unk_token)

table_marker = 'madeupword0000'
field_marker = 'madeupword0001'
value_marker = '=-=-=-=-=-'
asterisk_marker = 'madeupword0002'
table_marker_id = rt.convert_tokens_to_ids(table_marker)
field_marker_id = rt.convert_tokens_to_ids(field_marker)
value_marker_id = rt.convert_tokens_to_ids(value_marker)
asterisk_marker_id = rt.convert_tokens_to_ids(asterisk_marker)

text_field_marker = '=-=-'
number_field_marker = '=-=-=-=-'
time_field_marker = '--+'
boolean_field_marker = '||||'
other_field_marker = '=-=-=-=-=-=-=-=-'
text_field_marker_id = rt.convert_tokens_to_ids(text_field_marker)
number_field_marker_id = rt.convert_tokens_to_ids(number_field_marker)
time_field_marker_id = rt.convert_tokens_to_ids(time_field_marker)
boolean_field_marker_id = rt.convert_tokens_to_ids(boolean_field_marker)
other_field_marker_id = rt.convert_tokens_to_ids(other_field_marker)


typed_field_markers = [
    text_field_marker,
    number_field_marker,
    time_field_marker,
    boolean_field_marker,
    other_field_marker
]