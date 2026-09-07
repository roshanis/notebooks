"""Small, deterministic lexical retrieval over source-linked historical notes."""

import math
import re
from collections import Counter

STOPWORDS = set('a an the of in on for to and or is are was were did does do what how which have has that this it'.split())


def tokens(text):
    return [t for t in re.findall(r'[a-z0-9]+', text.lower()) if t not in STOPWORDS]


class Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.counts = [Counter(tokens(d['title'] + ' ' + d['text'])) for d in documents]
        self.df = Counter(t for count in self.counts for t in count)

    def search(self, query, limit=2):
        query_tokens = set(tokens(query))
        scored = []
        for document, counts in zip(self.documents, self.counts):
            score = sum((1 + math.log(counts[t])) * math.log(1 + len(self.documents) / self.df[t])
                        for t in query_tokens if counts[t])
            score /= math.sqrt(max(sum(counts.values()), 1))
            if score > 0:
                scored.append((score, document))
        return [d for _, d in sorted(scored, key=lambda x: (-x[0], x[1]['id']))[:limit]]
