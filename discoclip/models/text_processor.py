"""
Convert a sentence into an einsum expression and a list of tensor ids. 
The pipeline involves the following steps:
1. Tokenize the sentence.
2. Lemmatize the tokens.
3. Parse the sentence with BobcatParser or TreeParser.
4. Apply the MPSAnsatz to obtain a tensor network diagram.
5. Convert the diagram to an einsum expression and a list of symbols with their shapes.

Before training, we need to gather the symbols and their shapes from the training data.
The symbols will be used to initialize the tensors in the model.
"""

from abc import ABC, abstractmethod
from typing import Optional
from lambeq import CCGTree, TreeReader, TensorAnsatz, cups_reader

from ..utils.einsum import tn_to_einsum


class TextProcessor(ABC):
    """
    An abstract class for text processors that convert sentences into tensor network diagrams.
    Subclasses should implement the `parse` method.
    """

    @abstractmethod
    def parse(self, sentences: list[str], suppress_exceptions: bool = False):
        """
        Parse a list of sentences into a tensor network diagram.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def __call__(self, sentences: list[str], suppress_exceptions: bool = False):
        return self.parse(sentences, suppress_exceptions=suppress_exceptions)
    

class BobcatTextProcessor(TextProcessor):
    from lambeq import BobcatParser, TensorAnsatz, TreeReaderMode

    def __init__(self, ccg_parser: BobcatParser, ansatz: TensorAnsatz, 
                 tree_reader_mode: Optional[TreeReaderMode] = None):
        """
        Initialize the BobcatProcessor with a CCG parser and an ansatz.
        Args:
            ccg_parser: An instance of a CCG parser (e.g., BobcatParser).
            ansatz: An instance of a tensor ansatz (e.g., MPSAnsatz).
            tree_reader_mode: Optional mode for the tree reader. If
            provided, the diagram will be in the compact form.
        """
        self.ccg_parser = ccg_parser
        self.ansatz = ansatz
        self.tokenizer = Tokenizer()
        self.tree_reader_mode = tree_reader_mode
    
    def sentences2trees(self, sentences: list[str], suppress_exceptions: bool = False):
        """
        Convert a list of sentences into a list of CCG trees.
        """
        tokens = [self.tokenizer.tokenize(sent) for sent in sentences]
        lemmas = [self.tokenizer.lemmatize(tokens) for tokens in tokens]

        trees = self.ccg_parser.sentences2trees(tokens, tokenised=True, 
                                                suppress_exceptions=suppress_exceptions)
        lemma_trees = [
            self.lemmatize_tree(tree, lemma) if tree else None 
            for tree, lemma in zip(trees, lemmas)
        ]
        results =  {'tokens': tokens,
                'lemmas': lemmas,
                'trees': trees,
                'lemma_trees': lemma_trees
                }
        return results

    def parse(self, sentences: list[str], suppress_exceptions: bool = False):
        """
        Parse a list of sentences into a tensor network diagram.
        """
        results = self.sentences2trees(sentences, suppress_exceptions=suppress_exceptions)
        lemma_trees = results['lemma_trees']
        if self.tree_reader_mode is not None:
            diagrams = [TreeReader.tree2diagram(tree, mode=self.tree_reader_mode) 
                        if tree else None for tree in lemma_trees]
        else:
            diagrams = [tree.to_diagram() if tree else None for tree in lemma_trees]
        circuits = [self.ansatz(diagram) if diagram else None for diagram in diagrams]

        einsum_inputs = [tn_to_einsum(circuit) if circuit else None for circuit in circuits]


        results['diagrams'] = diagrams
        results['circuits'] = circuits
        results['einsum_inputs'] = einsum_inputs

        return results

    def lemmatize_tree(self, tree: CCGTree, lemmas: list):
        """
        Replace the text of the leaves in a CCG tree with the provided lemmas.
        Args:
            tree: A CCGTree instance.
            lemmas: A list of lemmas to replace the leaves' text.
        Returns:
            A new CCGTree with the leaves' text replaced by the lemmas.
        """
        tree = CCGTree.from_json(tree.to_json())
        def traverse(node: CCGTree, lemma_iter):
            if node.is_leaf:
                node._text = next(lemma_iter)
            else:
                for child in node.children:
                    traverse(child, lemma_iter)

        lemma_iter = iter(lemmas)
        traverse(tree, lemma_iter)
        return tree 
    

class CupsTextProcessor(TextProcessor):
    def __init__(self, ansatz: TensorAnsatz):
        """
        Initialize the CupsProcessor with a tensor ansatz.
        Args:
            ansatz: An instance of a tensor ansatz (e.g., MPSAnsatz).
        """
        self.ansatz = ansatz
        self.tokenizer = Tokenizer()
    
    def parse(self, sentences: list[str], suppress_exceptions: bool = False):
        """
        Parse a list of sentences into a tensor network diagram.
        """
        tokens = [self.tokenizer.tokenize(sent) for sent in sentences]
        lemmas = [self.tokenizer.lemmatize(tokens) for tokens in tokens]

        diagrams = cups_reader.sentences2diagrams(lemmas, tokenised=True)
        circuits = [self.ansatz(diagram) for diagram in diagrams]
        einsum_inputs = [tn_to_einsum(circuit) for circuit in circuits]

        results = {
            'tokens': tokens,
            'lemmas': lemmas,
            'diagrams': diagrams,
            'circuits': circuits,
            'einsum_inputs': einsum_inputs
        }
        return results

class VectorTextProcessor(TextProcessor):
    def __init__(self, ansatz: TensorAnsatz):
        """
        Initialize the VectorProcessor with a tensor ansatz.
        Args:
            ansatz: An instance of a tensor ansatz (e.g., MPSAnsatz).
        """
        self.ansatz = ansatz
        self.tokenizer = Tokenizer()
    
    def parse(self, sentences: list[str], suppress_exceptions: bool = False):
        """
        Parse a list of sentences into a tensor network diagram.
        Args:
            sentences: A list of sentences to parse.
            suppress_exceptions: Not used in this processor, but kept for compatibility.
        """
        tokens = [self.tokenizer.tokenize(sent) for sent in sentences]
        lemmas = [self.tokenizer.lemmatize(tokens) for tokens in tokens]

        results = {
            'tokens': tokens,
            'lemmas': lemmas,
        }
        return results