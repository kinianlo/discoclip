import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class SVODataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 text_transform,
                 progress = True):
        self.dataset = pd.read_csv(data_path)
        self.progress = progress

        required_columns = ['corrected_sentence', 'pos_image_id', 'neg_image_id']
        if not all(col in self.dataset.columns for col in required_columns):
            raise ValueError(f"SVODataset must contain the following columns: {required_columns}")

        if text_transform is not None:
            self.precompute_text(text_transform)

    def precompute_text(self, text_transform):
        self.text_map = {}
        symbol_size_set = set()

        captions = list(set(self.dataset['corrected_sentence'].tolist()))
        batch_size = 6

        for i in tqdm(range(0, len(captions), batch_size), disable=not self.progress):
            batch_captions = captions[i:i + batch_size]

            einsum_inputs = text_transform(batch_captions, suppress_exceptions=True)['einsum_inputs']
            for caption, einsum_input in zip(batch_captions, einsum_inputs):
                self.text_map[caption] = einsum_input
                if einsum_input is not None:
                    _, symbol_size_list = einsum_input
                    for symbol, size in symbol_size_list:
                        symbol_size_set.add((symbol, size))
        self.symbols, self.sizes = zip(*symbol_size_set) if symbol_size_set else ([], [])
                    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pos_image = self.dataset.iloc[idx]['pos_image_id']
        neg_image = self.dataset.iloc[idx]['neg_image_id']

        sentence = self.dataset.iloc[idx]['corrected_sentence']
        sentence = self.text_map.get(sentence, None)

        def remove_shape(einsum_input):
            if einsum_input is None:
                return None
            einsum_expr, symbol_size_list = einsum_input
            return (einsum_expr, [sym for sym, _ in symbol_size_list])
        sentence = remove_shape(sentence)

        return {
            "pos_image": pos_image,
            "neg_image": neg_image,
            "sentence": sentence,
            "index": idx
        }
    
    def state_dict(self):
        return {
            'text_map': self.text_map,
            'symbols': self.symbols,
            'sizes': self.sizes
        }
    
    def load_state_dict(self, state_dict):
        self.text_map = state_dict['text_map']
        self.symbols = state_dict['symbols']
        self.sizes = state_dict['sizes']

from discoclip.models import VectorTextProcessor

class SVOVectorDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 text_transform: VectorTextProcessor,
                 progress = True):
        self.dataset = pd.read_csv(data_path)
        self.progress = progress
       
        if text_transform is not None:
            self.precompute_text(text_transform)
        
    def precompute_text(self, text_transform: VectorTextProcessor):
        self.text_map = {}
        captions = list(set(self.dataset['true_caption'].tolist() + self.dataset['false_caption'].tolist()))
        self.vocab = set()
        
        for caption in tqdm(captions, disable=not self.progress):
            self.text_map[caption] = text_transform([caption], suppress_exceptions=True)['lemmas'][0]
            self.vocab.update(self.text_map[caption])
        self.vocab = list(self.vocab)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset.iloc[idx]['image_path']

        sentence = self.dataset.iloc[idx]['corrected_sentence']
        sentence = self.text_map.get(sentence, None)
        
        return {
            "image": image,
            "sentence": sentence,
            "index": idx
        }
    
    def state_dict(self):
        return {
            'text_map': self.text_map,
            'vocab': self.vocab
        }
    
    def load_state_dict(self, state_dict):
        self.text_map = state_dict['text_map']
        self.vocab = state_dict['vocab']
        

def svo_tn_collate_fn(batch):
    valid = [item['sentence'] is not None for item in batch]
    # print the number of invalid samples if any
    if not all(valid):
        print(f"Found {sum(not v for v in valid)} invalid samples")

    pos_images = [item['pos_image'] for item, v in zip(batch, valid) if v]
    neg_images = [item['neg_image'] for item, v in zip(batch, valid) if v]
    sentences = [item['sentence'] for item, v in zip(batch, valid) if v]
    indices = [item['index'] for item, v in zip(batch, valid) if v]

    return {
        "pos_images": pos_images,
        "neg_images": neg_images,
        "sentences": sentences,
        "indices": indices
    }
    
svo_vector_collate_fn = svo_tn_collate_fn