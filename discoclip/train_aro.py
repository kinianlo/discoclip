import argparse
import logging
import os

import mlflow
import torch
import torch.optim as optim
from torch import nn
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import trange

from discoclip.data import ARODataset, aro_tn_collate_fn
from discoclip.models import BobcatTextProcessor, EinsumModel, InfoNCE
from discoclip.utils import CachedBobcatParser, CustomMPSAnsatz, LookupEmbedding
from lambeq import AtomicType, Rewriter, Symbol
from lambeq.backend.tensor import Dim

torch.serialization.add_safe_globals([Symbol])

def parse_args():
    parser = argparse.ArgumentParser(description="Discoclip ARO Dataset Training Script")
    parser.add_argument('--train-data-path', type=str, required=True,
                        help='Path to the training data JSON file')

    parser.add_argument('--val-data-path', type=str, required=True,
                        help='Path to the validation data JSON file')

    parser.add_argument('--test-data-path', type=str, required=True,
                        help='Path to the test data JSON file')
                        
    parser.add_argument('--image-lookup-path', type=str, required=True,
                        help='Path to the image lookup file (e.g., CLIP embeddings)')

    parser.add_argument('--reader', type=str, default='bobcat',
                        choices=['bobcat', 'tree', 'tree_rule_only', 'tree_rule_type', 'tree_height', 'cups'],
                        help='Type of reader to use for text processing')

    parser.add_argument('--embedding-dim', type=int, default=512,
                        help='Dimension of the embeddings')

    parser.add_argument('--bond-dim', type=int, default=10,
                        help='Bond dimension for the MPS ansatz')

    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for the InfoNCE loss')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')

    parser.add_argument('--learning-rate', type=float, default=0.003,
                        help='Learning rate for the optimizer')

    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay for the optimizer')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train the model')

    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    parser.add_argument('--hard-neg-loss-weight', type=float, default=10.0,
                        help='Weight for the hard negative loss component')

    parser.add_argument('--log-path', type=str, default='logs',
                        help='Path to save training logs')

    parser.add_argument('--checkpoint-path', type=str, default='checkpoints',
                        help='Path to save model checkpoints')

    parser.add_argument('--mlflow-uri', type=str, default='file:./mlruns',
                        help='URI for MLflow tracking server')

    parser.add_argument('--mlflow-experiment', type=str, default='discoclip_aro',
                        help='Name of the MLflow experiment')

    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to run the training on (cpu, cuda, mps)')
    return parser.parse_args()


def get_aro_dataset(
    data_path: str,
    reader: str,
    dim: int,
    bond_dim: int,
    progress: bool = True
    ) -> ARODataset:

    ansatz = CustomMPSAnsatz(
        {AtomicType.SENTENCE: Dim(dim),
         AtomicType.NOUN: Dim(dim),
         AtomicType.PREPOSITIONAL_PHRASE: Dim(dim)},
         bond_dim=bond_dim,
    )

    if reader == 'bobcat':
        rules = ['auxiliary', 'connector', 'determiner', 'postadverb', 'preadverb',
                'prepositional_phrase', 'coordination', 'object_rel_pronoun', 'subject_rel_pronoun']
        rewriter = Rewriter(rules)
        bobcat_parser = CachedBobcatParser()
        text_transform = BobcatTextProcessor(
            ccg_parser=bobcat_parser,
            ansatz=ansatz,
            rewriter=rewriter,
        )
    elif reader.startswith('tree'):
        from lambeq import TreeReaderMode
        bobcat_parser = CachedBobcatParser()
        if reader == 'tree':
            tree_reader_mode = TreeReaderMode.NO_TYPE
        elif reader == 'tree_rule_only':
            tree_reader_mode = TreeReaderMode.RULE_ONLY
        elif reader == 'tree_rule_type':
            tree_reader_mode = TreeReaderMode.RULE_TYPE
        elif reader == 'tree_height':
            tree_reader_mode = TreeReaderMode.HEIGHT
        else:
            raise ValueError(f"Unknown tree reader type: {reader}")

        text_transform = BobcatTextProcessor(
            ccg_parser=bobcat_parser,
            ansatz=ansatz,
            rewriter=None,
            tree_reader_mode=tree_reader_mode
        )
    elif reader == 'cups':
        from discoclip.models import CupsTextProcessor
        text_transform = CupsTextProcessor(ansatz=ansatz)

    return ARODataset(
        data_path=data_path,
        text_transform=text_transform,
        progress=progress
    )

def setup_logger(log_path):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def train_epoch(model, image_model, dataloader,
                contrastive_criterion,
                hard_neg_criterion,
                optimizer,
                hard_neg_loss_weight=0,
                device='cpu'):
    """
    Train the model for one epoch.
    """
    model.train()
    image_model.train()

    metrics = {
        'loss': 0.0,
        'contrastive_loss': 0.0,
        'contrastive_acc': 0.0,
        'hard_neg_loss': 0.0,
        'hard_neg_acc': 0.0,
        'hard_neg_draw': 0.0,
        'text_embedding_mean_norm': 0.0
    }

    total_samples = 0

    for batch in dataloader:
        batch_size = len(batch['images'])
        optimizer.zero_grad()

        images = batch['images']
        true_captions = batch['true_captions']

        with torch.no_grad():
            image_embeddings = image_model(images).to(device)
        text_embeddings = model(true_captions)

        metrics['text_embedding_mean_norm'] += text_embeddings.norm(dim=-1).mean().item()

        infonce_loss, infonce_acc = contrastive_criterion(image_embeddings, text_embeddings)

        false_captions = batch['false_captions']
        hard_neg_text_embeddings = model(false_captions)
        pos_sim = cosine_similarity(text_embeddings, image_embeddings, dim=-1)
        neg_sim = cosine_similarity(hard_neg_text_embeddings, image_embeddings, dim=-1)
        pred = (pos_sim > neg_sim).float()
        hard_neg_loss = hard_neg_criterion(pred, torch.ones_like(pred))

        loss = infonce_loss + hard_neg_loss_weight * hard_neg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        metrics['loss'] += loss.item() * batch_size
        metrics['contrastive_loss'] += infonce_loss.item() * batch_size
        metrics['contrastive_acc'] += infonce_acc.item() * batch_size
        metrics['hard_neg_loss'] += hard_neg_loss.item() * batch_size
        metrics['hard_neg_acc'] += (pos_sim > neg_sim).float().mean().item() * batch_size
        metrics['hard_neg_draw'] += (pos_sim == neg_sim).float().mean().item() * batch_size
        total_samples += batch_size

    for key in metrics:
        metrics[key] /= total_samples
    return metrics

def evaluate_model(model, image_model, dataloader,
                   contrastive_criterion, 
                   hard_neg_criterion=None, 
                   hard_neg_loss_weight=0, 
                   device='cpu'):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    image_model.eval()

    metrics = {
        'loss': 0.0,
        'contrastive_loss': 0.0,
        'contrastive_acc': 0.0,
        'hard_neg_loss': 0.0,
        'hard_neg_acc': 0.0,
        'hard_neg_draw': 0.0
    }

    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_size = len(batch['images'])
            images = batch['images']
            true_captions = batch['true_captions']

            image_embeddings = image_model(images).to(device)
            text_embeddings = model(true_captions)

            infonce_loss, infonce_acc = contrastive_criterion(image_embeddings, text_embeddings)

            false_captions = batch['false_captions']
            hard_neg_text_embeddings = model(false_captions)
            pos_sim = cosine_similarity(text_embeddings, image_embeddings, dim=-1)
            neg_sim = cosine_similarity(hard_neg_text_embeddings, image_embeddings, dim=-1)
            pred = (pos_sim > neg_sim).float()
            hard_neg_loss = hard_neg_criterion(pred, torch.ones_like(pred)) 

            loss = infonce_loss + hard_neg_loss_weight * hard_neg_loss

            metrics['contrastive_loss'] += infonce_loss.item() * batch_size
            metrics['contrastive_acc'] += infonce_acc.item() * batch_size
            metrics['loss'] += loss.item() * batch_size
            metrics['hard_neg_loss'] += hard_neg_loss.item() * batch_size
            metrics['hard_neg_acc'] += (pos_sim > neg_sim).float().mean().item() * batch_size
            metrics['hard_neg_draw'] += (pos_sim == neg_sim).float().mean().item() * batch_size
            total_samples += batch_size

    for key in metrics:
        metrics[key] /= total_samples
    return metrics

def train_model(args):
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(vars(args))

        logger = setup_logger(os.path.join(args.log_path, f"train_{run.info.run_id}.log"))
        logger.info(f"Running experiment: {args.mlflow_experiment}, run ID: {run.info.run_id}, run name: {run.info.run_name}")

        # Get datasets
        train_ds = get_aro_dataset(
            data_path=args.train_data_path,
            reader=args.reader,
            dim=args.embedding_dim,
            bond_dim=args.bond_dim,
            progress=True
        )

        val_ds = get_aro_dataset(
            data_path=args.val_data_path,
            reader=args.reader,
            dim=args.embedding_dim,
            bond_dim=args.bond_dim,
            progress=True
        )

        test_ds = get_aro_dataset(
            data_path=args.test_data_path,
            reader=args.reader,
            dim=args.embedding_dim,
            bond_dim=args.bond_dim,
            progress=True
        )

        # Init dataloaders
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=aro_tn_collate_fn
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=aro_tn_collate_fn
        )
        
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=aro_tn_collate_fn
        )

        # Initialize models
        symbols = train_ds.symbols + val_ds.symbols + test_ds.symbols
        sizes = train_ds.sizes + val_ds.sizes + test_ds.sizes
        model = EinsumModel(symbols, sizes).to(args.device)

        image_model = LookupEmbedding.load_from_checkpoint(args.image_lookup_path)
        image_model = image_model.to(args.device)

        # Define optimizer and loss functions
        infonce = InfoNCE()
        bce = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), 
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)

        best_val_hard_neg_loss = float('inf')

        for epoch in trange(1, args.epochs + 1, desc="Training Epochs"):
            logger.info(f"Starting epoch {epoch}/{args.epochs}")

            # Train
            train_metrics = train_epoch(model, image_model, train_loader,
                                infonce, bce, optimizer,
                                hard_neg_loss_weight=args.hard_neg_loss_weight,
                                device=args.device)
            mlflow.log_metrics({f'train/{key}': value for key, value in train_metrics.items()}, step=epoch)

            # Evaluate
            val_metrics = evaluate_model(model, image_model, val_loader,
                                            infonce, bce, 
                                            hard_neg_loss_weight=args.hard_neg_loss_weight,
                                            device=args.device)

            mlflow.log_metrics({f'val/{key}': value for key, value in val_metrics.items()}, step=epoch)
            logger.info(f"Epoch {epoch}/{args.epochs} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val Acc: {val_metrics['hard_neg_acc']:.4f}")

            # Save best model checkpoint
            if val_metrics['hard_neg_loss'] < best_val_hard_neg_loss:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_metrics': val_metrics,
                    'args': vars(args)
                }
                checkpoint_path = os.path.join(args.checkpoint_path, f"{run.info.run_id}/best_model.pt")
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(checkpoint, checkpoint_path)
                mlflow.log_artifact(checkpoint_path)
                logger.info(f"Saved best model checkpoint to {checkpoint_path}")

        # Final evaluation on test set
        best_model_path = os.path.join(args.checkpoint_path, f"{run.info.run_id}/best_model.pt")
        best_checkpoint = torch.load(best_model_path, map_location=args.device)
        best_model = EinsumModel(symbols, sizes).to(args.device)
        best_model.load_state_dict(best_checkpoint['model_state_dict'])

        test_metrics = evaluate_model(best_model, image_model, test_loader,
                                      infonce, bce, 
                                      hard_neg_loss_weight=args.hard_neg_loss_weight,
                                      device=args.device)
        mlflow.log_metrics({f'test/{key}': value for key, value in test_metrics.items()})
        logger.info(f"Testing on test set with best model from epoch {best_checkpoint['epoch']}")
        logger.info(f"Test Loss: {test_metrics['loss']:.4f}, "
                    f"Test Acc: {test_metrics['hard_neg_acc']:.4f}")
        logger.info("Training complete.")
        mlflow.log_artifact(os.path.join(args.log_path, f"train_{run.info.run_id}.log"))

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    train_model(args)
