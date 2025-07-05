import re
import math
from urllib.parse import urlparse, parse_qs
from collections import Counter
import numpy as np

def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count / lns * math.log2(count / lns) for count in p.values() if count)

def extract_features(url):
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query_string = parsed.query
    tokens = re.split(r'\W+', query_string)
    token_lengths = [len(t) for t in tokens if t]
    token_hist = Counter(tokens)

    common_tokens = ['select', 'or', 'and', 'script', 'alert']
    hist_features = [token_hist.get(tok, 0) for tok in common_tokens]
    e = entropy(query_string)

    special_chars = ['=', '&', '\'', '"', '<', '>']
    special_count = sum(url.count(c) for c in special_chars)
    url_length = len(url)
    num_params = len(query)

    # Risky keywords (binary flags)
    risky_keywords = ['drop', 'insert', 'update', 'script', 'alert']
    risky_flags = [int(keyword in url.lower()) for keyword in risky_keywords]

    # NEW: Specific injection-type signals
    has_script_tag = int(bool(re.search(r"<script.*?>", url, re.IGNORECASE)))
    has_img_onerror = int(bool(re.search(r"<img[^>]+onerror=", url, re.IGNORECASE)))
    has_iframe = int("<iframe" in url.lower())
    has_union = int("union" in url.lower())
    has_double_dash = int("--" in url)
    has_single_quote = int("'" in url)
    has_encoded_script = int("%3Cscript" in url.lower())  # Encoded <
    has_encoded_single_quote = int("%27" in url.lower())  # Encoded '

    return np.array([
        len(tokens),
        sum(token_lengths),
        int(bool(re.search(r"\b(or|and)\b.*=", query_string))),  # logic operators
        int(bool(re.search(r"alert|onerror|prompt|confirm", url, re.IGNORECASE))),
        e,
        *hist_features,
        special_count,
        url_length,
        num_params,
        *risky_flags,
        has_script_tag,
        has_img_onerror,
        has_iframe,
        has_union,
        has_double_dash,
        has_single_quote,
        has_encoded_script,
        has_encoded_single_quote
    ])
