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

from discoclip.utils.einsum import tn_to_einsum
from discoclip.models.tokenizer import Tokenizer


class TextProcessor(ABC):
    """
    An abstract class for text processors that convert sentences into tensor network diagrams.
    Subclasses should implement the `parse` method.
    """

    @abstractmethod
    def parse(self, sentences: list[str], suppress_exceptions: bool = False, 
              return_details: bool = False):
        """
        Parse a list of sentences into a tensor network diagram.
        Args:
            sentences: A list of sentences to parse.
            suppress_exceptions: Whether to suppress exceptions during parsing.
            return_details: Whether to return detailed intermediate results.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def __call__(self, sentences: list[str], suppress_exceptions: bool = False,
                return_details: bool = False):
        return self.parse(sentences, suppress_exceptions=suppress_exceptions,
                         return_details=return_details)
    

class BobcatTextProcessor(TextProcessor):
    from lambeq import BobcatParser, TensorAnsatz, TreeReaderMode, Rewriter

    def __init__(self, ccg_parser: BobcatParser, ansatz: TensorAnsatz,
                 rewriter: Optional[Rewriter] = None,
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
        self.rewriter = rewriter
        self.tree_reader_mode = tree_reader_mode
    
    def sentences2trees(self, sentences: list[str], 
                        suppress_exceptions: bool = False,
                        return_details: bool = False):
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

        if return_details:
            return {'tokens': tokens,
                    'lemmas': lemmas,
                    'trees': trees,
                    'lemma_trees': lemma_trees}
        else:
            del tokens, lemmas, trees  # Free memory
            return {'lemma_trees': lemma_trees}

    def parse(self, sentences: list[str], suppress_exceptions: bool = False,
              return_details: bool = False):
        """
        Parse a list of sentences into a tensor network diagram.
        """
        results = self.sentences2trees(sentences, suppress_exceptions=suppress_exceptions,
                                       return_details=return_details)

        lemma_trees = results['lemma_trees']
        if self.tree_reader_mode is not None:
            diagrams = [TreeReader.tree2diagram(tree, mode=self.tree_reader_mode) 
                        if tree else None for tree in lemma_trees]
        else:
            diagrams = [tree.to_diagram() if tree else None for tree in lemma_trees]

        if self.rewriter is not None:
            rewritten_diagrams = [self.rewriter(diagram).remove_snakes() if diagram else None for diagram in diagrams]
        else:
            rewritten_diagrams = diagrams  # Use diagrams directly if no rewriter

        circuits = [self.ansatz(diagram) if diagram else None for diagram in rewritten_diagrams]

        einsum_inputs = [tn_to_einsum(circuit) if circuit else None for circuit in circuits]

        # Create a new results dictionary with only what we need
        filtered_results = {}
        filtered_results['einsum_inputs'] = einsum_inputs
        
        if return_details:
            filtered_results['lemma_trees'] = lemma_trees
            filtered_results['diagrams'] = diagrams
            filtered_results['rewritten_diagrams'] = rewritten_diagrams
            filtered_results['circuits'] = circuits
            # Copy any other fields from the original results dictionary
            for key in results:
                if key != 'lemma_trees':  # Already added this above
                    filtered_results[key] = results[key]
        
        # Explicitly delete all large objects if not returning them
        del results
        del einsum_inputs
        del diagrams
        del rewritten_diagrams
        del circuits
        del lemma_trees  # Free memory
        
        return filtered_results

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
    
    def parse(self, sentences: list[str], suppress_exceptions: bool = False,
              return_details: bool = False):
        """
        Parse a list of sentences into a tensor network diagram.
        Args:
            sentences: A list of sentences to parse.
            suppress_exceptions: Whether to suppress exceptions during parsing.
            return_details: Whether to return detailed intermediate results.
        """
        tokens = [self.tokenizer.tokenize(sent) for sent in sentences]
        lemmas = [self.tokenizer.lemmatize(tokens) for tokens in tokens]

        diagrams = cups_reader.sentences2diagrams(lemmas, tokenised=True)
        circuits = [self.ansatz(diagram) for diagram in diagrams]
        einsum_inputs = [tn_to_einsum(circuit) for circuit in circuits]

        results = {'einsum_inputs': einsum_inputs}
        
        if return_details:
            results.update({
                'tokens': tokens,
                'lemmas': lemmas,
                'diagrams': diagrams,
                'circuits': circuits
            })
        else:
            # Free memory for large objects
            del tokens, lemmas, diagrams, circuits
            
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
    
    def parse(self, sentences: list[str], suppress_exceptions: bool = False,
              return_details: bool = False):
        """
        Parse a list of sentences into a tensor network diagram.
        Args:
            sentences: A list of sentences to parse.
            suppress_exceptions: Not used in this processor, but kept for compatibility.
            return_details: Whether to return detailed intermediate results.
        """
        tokens = [self.tokenizer.tokenize(sent) for sent in sentences]
        lemmas = [self.tokenizer.lemmatize(tokens) for tokens in tokens]

        # For VectorTextProcessor, minimal results are the same as detailed results
        # since it doesn't have diagrams or circuits
        results = {
            'lemmas': lemmas,
        }
        
        if return_details:
            results['tokens'] = tokens
        else:
            del tokens  # Free memory when not needed
            
        return results