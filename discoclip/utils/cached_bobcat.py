import diskcache
import os

from lambeq import BobcatParser
from lambeq.core.utils import SentenceBatchType

class CachedBobcatParser(BobcatParser):
    def __init__(self, *args, cache_path="~/.cache/lambeq/bobcat/diskcache", **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_path = os.path.expanduser(cache_path)
        self._cache = diskcache.Cache(self._cache_path)

    def sentences2trees(self,
        sentences: SentenceBatchType,
        tokenised: bool = False,
        suppress_exceptions: bool = False,
        verbose: str | None = None):

        # Split sentences into batches if they exceed the batch size
        # This is to avoid computing the same sentence multiple times
        batch_size = getattr(self, 'batch_size', 16)
        if len(sentences) > batch_size:
            results = []
            for i in range(0, len(sentences), batch_size):
                batch_results = self.sentences2trees(
                    sentences[i:i + batch_size],
                    tokenised=tokenised,
                    suppress_exceptions=suppress_exceptions,
                    verbose=verbose
                )
                results.extend(batch_results)
            return results

        results = [None] * len(sentences)
        uncached_sentences = []
        
        for i, sent in enumerate(sentences):
            key = str((sent, tokenised, suppress_exceptions))
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_sentences.append((i, sent))
        
        if uncached_sentences:
            uncached_results = super().sentences2trees(
                [sent for _, sent in uncached_sentences],
                tokenised=tokenised,
                suppress_exceptions=suppress_exceptions, 
                verbose=verbose
            )
            
            for (i, sent), result in zip(uncached_sentences, uncached_results):
                key = str((sent, tokenised, suppress_exceptions))
                self._cache[key] = result
                results[i] = result

        return results