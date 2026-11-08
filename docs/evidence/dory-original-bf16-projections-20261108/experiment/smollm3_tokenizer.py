"""Freeze only the experiment's native template clock for reproducible prompts.

SmolLM3's unchanged chat template prints today's date. Supplying its Jinja
strftime_now callable preserves the native template while keeping paired runs
identical across midnight. This affects local pretokenized benchmark requests,
not the server's general chat API or any installed tokenizer source.
"""
from datetime import datetime, timezone
from functools import wraps

TEMPLATE_DATE_UTC = '2026-10-31'
_DATE = datetime(2026, 10, 31, tzinfo=timezone.utc)


def fixed_strftime(format_string):
    return _DATE.strftime(format_string)


def pin_template_date(tokenizer):
    if getattr(tokenizer, '_trn1_template_date', None) == TEMPLATE_DATE_UTC:
        return tokenizer
    original = tokenizer.apply_chat_template

    @wraps(original)
    def apply_chat_template(*args, **kwargs):
        kwargs['strftime_now'] = fixed_strftime
        return original(*args, **kwargs)

    tokenizer.apply_chat_template = apply_chat_template
    tokenizer._trn1_template_date = TEMPLATE_DATE_UTC
    return tokenizer
