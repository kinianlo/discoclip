from nltk import pos_tag
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class Tokenizer:
    """
    A simple tokenizer and lemmatizer using NLTK's Treebank
    tokenizer and WordNet lemmatizer.
    """
    def __init__(self):
        self.word_tokenizer = TreebankWordTokenizer()
        self.lemmatizer = WordNetLemmatizer()

    def tokenize(self, sentence):
        tokens = self.word_tokenizer.tokenize(sentence)
        return tokens
    
    def lemmatize(self, tokens):
        pos_tags = pos_tag(tokens)
        lemmas = [
            self.lemmatizer.lemmatize(token.lower(), pos=self._to_wordnet_pos(tag))
            for token, tag in pos_tags
        ]
        return lemmas

    def _to_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def __call__(self, sentence):
        """
        Tokenize and lemmatize a sentence.
        """
        tokens = self.tokenize(sentence)
        lemmas = self.lemmatize(tokens)
        return lemmas