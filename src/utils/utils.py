"""
Utility functions.
"""
import collections
import datetime
import functools
import inspect
import random
import re
import string
import warnings

import src.utils.trans.bert_utils as bu
import src.utils.trans.bert_cased_utils as bcu
import src.utils.trans.roberta_utils as ru
# import src.utils.trans.table_bert_utils as tbu

from nltk.corpus import stopwords
try:
    _stopwords = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    _stopwords = set(stopwords.words('english'))


string_types = (type(b''), type(u''))


SEQ2SEQ = 0
SEQ2SEQ_PG = 1
SQLOVA = 2
RATSQL = 4
TRANSFORMER = 5
TRANSFORMER_PG = 6
VASE = 7

SQLOVA_POINT_PT = 100
SQLOVA_ML_PT = 101


model_index = {
    'seq2seq': SEQ2SEQ,
    'seq2seq.pg': SEQ2SEQ_PG,
    'sqlova': SQLOVA,
    'ratsql': RATSQL,
    'transformer': TRANSFORMER,
    'transformer.pg': TRANSFORMER_PG,
    'sqlova.pt': SQLOVA_POINT_PT,
    'sqlova.ml.pt': SQLOVA_ML_PT,
    'vase': VASE
}


# --- string utilities --- #

def is_number(s):
    try:
        float(s.replace(',', ''))
        return True
    except:
        return False


def is_stopword(s):
    return s.strip() in _stopwords


def is_common_db_term(s):
    return s.strip() in ['id']


def to_string(v):
    if isinstance(v, bytes):
        try:
            s = v.decode('utf-8')
        except UnicodeDecodeError:
            s = v.decode('latin-1')
    else:
        s = str(v)
    return s


def encode_str_list(l, encoding):
    return [x.encode(encoding) for x in l]


def list_to_hist(l):
    hist = collections.defaultdict(int)
    for x in l:
        hist[x] += 1
    return hist


def remove_parentheses_str(s):
    return re.sub(r'\([^)]*\)', '', s).strip()


def restore_feature_case(features, s):
    tokens = []
    i = 0
    for feat in features:
        if feat.endswith('##'):
            feat_ = feat[:-2]
        elif feat.startswith('##'):
            feat_ = feat[2:]
        else:
            feat_ = feat
        while not s[i].strip():
            i += 1
        token = s[i:i+len(feat_)]
        i = i + len(feat_)
        if feat.endswith('##'):
            token += '##'
        if feat.startswith('##'):
            token = '##' + token
        # assert(token.lower() == feat)
        assert(len(token) == len(feat))
        tokens.append(token)
    return tokens


def get_sub_token_ids(question_tokens, span_ids, tu):
    st, ed = span_ids
    prefix_tokens = question_tokens[:st]
    prefix = tu.tokenizer.convert_tokens_to_string(prefix_tokens)
    prefix_sub_tokens = tu.tokenizer.tokenize(prefix)

    span_tokens = question_tokens[st:ed]
    span = tu.tokenizer.convert_tokens_to_string(span_tokens)
    span_sub_tokens = tu.tokenizer.tokenize(span)

    return len(prefix_sub_tokens), len(prefix_sub_tokens) + len(span_sub_tokens)


def get_trans_utils(args):
    if args.pretrained_transformer.startswith('bert-') and args.pretrained_transformer.endswith('-uncased'):
        return bu
    elif args.pretrained_transformer.startswith('bert-') and args.pretrained_transformer.endswith('-cased'):
        return bcu
    elif args.pretrained_transformer.startswith('roberta-'):
        return ru
    elif args.pretrained_transformer in ['table-bert']:
        return tbu
    elif args.pretrained_transformer == '':
        return None
    else:
        raise NotImplementedError


def get_random_tag(k=6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=k))


def get_time_tag():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')


def strip_quotes(s):
    start = 0
    while start < len(s):
        if s[start] in ['"', '\'']:
            start += 1
        else:
            break
    end = len(s)
    while end > start:
        if s[end-1] in ['"', '\'']:
            end -= 1
        else:
            break
    if start == end:
        return ''
    else:
        return s[start:end]


# --- other utilities --- #

def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    if isinstance(reason, string_types):

        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))
