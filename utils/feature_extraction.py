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
    tokens = re.split(r'\W+', parsed.query)
    token_lengths = [len(t) for t in tokens if t]
    token_hist = Counter(tokens)
    common_tokens = ['select', 'or', 'and', 'script', 'alert']

    hist_features = [token_hist.get(tok, 0) for tok in common_tokens]
    e = entropy(parsed.query)

    special_chars = ['=', '&', '\'', '"', '<', '>']
    special_count = sum(url.count(c) for c in special_chars)
    url_length = len(url)
    num_params = len(query)
    risky_keywords = ['drop', 'insert', 'update', 'script', 'alert']
    risky_flags = [int(keyword in url.lower()) for keyword in risky_keywords]

    return np.array([
        len(tokens),
        sum(token_lengths),
        int(bool(re.search(r"\b(or|and)\b.*=", parsed.query))),
        int(bool(re.search(r"<script>|alert|onerror", url, re.IGNORECASE))),
        e,
        *hist_features,
        special_count,           
        url_length,              
        num_params,              
        *risky_flags
    ])
