"""
Encoder DB content.
"""

import difflib
from mo_future import string_types
from src.utils.utils import deprecated


class Match(object):
    def __init__(self, start, size):
        self.start = start
        self.size = size


def is_span_separator(c):
    return c in '\'"()`,.?! '


def split(s):
    return [c.lower() for c in s.strip()]


def get_effecitve_match_source(s, start, end):
    _start = -1

    for i in range(start, start - 2, -1):
        if i < 0:
            _start = i + 1
            break
        if is_span_separator(s[i]):
            _start = i
            break

    if _start < 0:
        return None

    _end = -1
    for i in range(end - 1, end + 3):
        if i >= len(s):
            _end = i - 1
            break
        if is_span_separator(s[i]):
            _end = i
            break

    if _end < 0:
        return None

    while(_start < len(s) and is_span_separator(s[_start])):
        _start += 1
    while(_end >= 0 and is_span_separator(s[_end])):
        _end -= 1

    return Match(_start, _end - _start + 1)


def get_matched_entries(s, field_values, m_theta=0.8, s_theta=0.5, k=1):
    if not field_values:
        return None

    if isinstance(s, str):
        n_grams = split(s)
    else:
        n_grams = s

    matched = dict()
    for field_value in field_values:
        if not isinstance(field_value, string_types):
            continue
        fv_tokens = split(field_value)
        sm = difflib.SequenceMatcher(None, n_grams, fv_tokens)
        match = sm.find_longest_match(0, len(n_grams), 0, len(fv_tokens))
        if match.size > 0:
            source_match = get_effecitve_match_source(n_grams, match.a, match.a + match.size)
            if source_match and source_match.size > 1:
                match_str = field_value[match.b:match.b + match.size]
                if match_str.strip():
                    s_match_score = match.size / source_match.size
                    match_score = match.size / len(fv_tokens)
                    if match_score >= m_theta and s_match_score > s_theta:
                        if field_value.isupper() and match_score * s_match_score < 1:
                            pass
                        else:
                            matched[field_value] = (match_score, s_match_score, match.size)

    # debug_span = 'ny'
    # debug_s = 'What are the number of votes from state'
    # if debug_span in [str(x).lower() for x in field_values] and s.startswith(debug_s):
    #     import pdb
    #     pdb.set_trace()
    if not matched:
        return None
    else:
        return sorted(matched.items(), key=lambda x:(1e16 * x[1][0] + 1e8 * x[1][1] + x[1][2]), reverse=True)[:k]


@deprecated
def split_old(s):
    return [' '] + [c.lower() for c in s.strip()] + [' ']


@deprecated
def source_match_score(s, start, end):
    _start = -1

    for i in range(start, start-2, -1):
        if i < 0:
            _start = i + 1
            break
        if not s[i].strip():
            _start = i
            break

    if _start < 0:
        return 0

    _end = -1
    for i in range(end-1, end + 3):
        if i >= len(s):
            _end = i - 1
            break
        if not s[i].strip() or s[i] == ',':
            _end = i
            break

    if _end < 0:
        return 0

    fuzzy_match_size = _end + 1 - _start
    fuzzy_match_score = (end - start) / fuzzy_match_size
    return fuzzy_match_score


@deprecated
def get_matched_entries_old(s, field_values, threshold=0.5, k=1):
    if not field_values:
        return None

    if isinstance(s, str):
        n_grams = split_old(s)
    else:
        n_grams = s

    matched = dict()
    for field_value in field_values:
        if not isinstance(field_value, string_types):
            return None
        fv_tokens = split_old(field_value)
        s = difflib.SequenceMatcher(None, n_grams, fv_tokens)
        match = s.find_longest_match(0, len(n_grams), 0, len(fv_tokens))
        if match.size > 0 and match.b == 0:
            match_str = field_value[match.b:match.b + match.size]
            if match_str.strip():
                match_score = match.size / len(fv_tokens)
                s_match_sore = source_match_score(n_grams, match.a, match.a + match.size)
                if match_score >= threshold and s_match_sore > 0:
                    if field_value.isupper() and match_score * s_match_sore < 1:
                        pass
                    else:
                        matched[field_value] = (match_score, s_match_sore, match.size)

    if not matched:
        return None
    else:
        return sorted(matched.items(), key=lambda x:x[1], reverse=True)[:k]


