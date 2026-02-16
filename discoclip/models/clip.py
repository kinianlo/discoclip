import torch
import torch.nn as nn

class CustomTextCLIP(nn.Module):
    def __init__(self, text_layers=6, text_width=256, text_heads=4, embed_dim=512, vocab_size=49408):
        super().__init__()
        
        # Store hyperparameters
        self.text_layers = text_layers
        self.text_width = text_width
        self.text_heads = text_heads
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, text_width)
        self.positional_embedding = nn.Parameter(torch.empty(77, text_width))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_width, nhead=text_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=text_layers)
        
        # Final projection to match the image embedding space
        self.text_projection = nn.Parameter(torch.empty(text_width, self.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # log(14.28) approx
        
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=self.text_width ** -0.5)

    def encode_text(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding[None, :, :]
        x = self.transformer(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image_features, text):
        text_features = self.encode_text(text)
        
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        logits = self.logit_scale.exp() * image_features @ text_features.t()
        return logits, logits.t()
    
    def save_to_checkpoint(self, path: str):
        """
        Save model weights and architecture hyperparameters to a checkpoint file.
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'text_layers': self.text_layers,
            'text_width': self.text_width,
            'text_heads': self.text_heads,
            'embed_dim': self.embed_dim,
            'vocab_size': self.vocab_size
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load_from_checkpoint(cls, path: str, map_location=None):
        """
        Load a model from a checkpoint file.
        Returns a CustomTextCLIP with the saved architecture and weights.
        """
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(
            text_layers=checkpoint['text_layers'],
            text_width=checkpoint['text_width'],
            text_heads=checkpoint['text_heads'],
            embed_dim=checkpoint['embed_dim'],
            vocab_size=checkpoint['vocab_size']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
class SimpleWordTokenizer:
    def __init__(self, sentences=None, context_length=77):
        self.sot = "<|startoftext|>"
        self.eot = "<|endoftext|>"
        self.pad = "<|pad|>"
        
        self.vocab = {self.pad: 0, self.sot: 1, self.eot: 2}
        self.context_length = context_length
        
        if sentences:
            self.build_vocab(sentences)

    def _clean(self, text):
        """Lower case and strip the last period only."""
        text = text.lower().strip().strip('.')
        return text

    def build_vocab(self, sentences):
        """Gather all unique words from a list of sentences."""
        for s in sentences:
            clean_s = self._clean(s)
            words = clean_s.split()
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        
        self.decoder = {v: k for k, v in self.vocab.items()}

    def __call__(self, texts, context_length=None):
        """Accepts a string or list of strings and returns a torch tensor."""
        if isinstance(texts, str):
            texts = [texts]
        
        length = context_length or self.context_length
        result = torch.zeros(len(texts), length, dtype=torch.long)

        for i, text in enumerate(texts):
            clean_text = self._clean(text)
            words = clean_text.split()
            
            tokens = [self.vocab[self.sot]]
            
            tokens += [self.vocab.get(word, 0) for word in words]
            tokens.append(self.vocab[self.eot])
            
            tokens = tokens[:length]
            result[i, :len(tokens)] = torch.tensor(tokens)
            
        return result

    def tokenize(self, texts) -> list[str]:
        return self._clean(texts).split()
    
    @property
    def vocab_size(self):
        return len(self.vocab)
    
    def save_to_checkpoint(self, path: str):
        """
        Save the tokenizer vocabulary and settings to a checkpoint file.
        """
        checkpoint = {
            'vocab': self.vocab,
            'context_length': self.context_length,
            'sot': self.sot,
            'eot': self.eot,
            'pad': self.pad
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load_from_checkpoint(cls, path: str):
        """
        Load a tokenizer from a checkpoint file.
        Returns a SimpleWordTokenizer with the saved vocabulary.
        """
        checkpoint = torch.load(path)
        tokenizer = cls(sentences=None, context_length=checkpoint['context_length'])
        tokenizer.sot = checkpoint['sot']
        tokenizer.eot = checkpoint['eot']
        tokenizer.pad = checkpoint['pad']
        tokenizer.vocab = checkpoint['vocab']
        tokenizer.decoder = {v: k for k, v in tokenizer.vocab.items()}
        return tokenizer
