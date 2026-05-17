#!/usr/bin/env python3
# Language Model Training Benchmark

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)

print('=' * 70)
print('Language Model Training Benchmark')
print('Device:', DEVICE)
print('PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU memory: {:.1f} GB'.format(torch.cuda.get_device_properties(0).total_memory / 1e9))
print('=' * 70)


@dataclass
class GPTConfig:
    vocab_size: int = 32000
    block_size: int = 256
    n_embd: int = 384
    n_layer: int = 4
    n_head: int = 6
    dropout: float = 0.0
    bias: bool = False


class CasualSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            att = att.masked_fill(mask[:, :, :, :] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CasualSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        mlp_hidden = config.n_embd * 4
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)).view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
            'lm_head': nn.Linear(config.n_embd, config.vocab_size, bias=False)
        })
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, pad_token_id=0):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.transformer['wte'](idx)
        pos_emb = self.transformer['wpe'](pos)
        x = tok_emb + pos_emb
        mask = self.causal_mask[:, :, :T, :T]

        for block in self.transformer['h']:
            x = block(x, mask)

        x = self.transformer['ln_f'](x)
        logits = self.transformer['lm_head'](x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token_id)

        return logits, loss


class RepoNativeGPTBaseline(SimpleGPT):
    """Repo-native GPT baseline used as the non-Clifford comparison."""
    pass


# Real CliffordAttractor from attractor.models.clifford_attractor
class CliffordAttractorLM(nn.Module):
    """Wrapper for CliffordAttractor to work with language model benchmark."""

    def __init__(
        self,
        config: GPTConfig,
        algebra_config=None,
        variant: str = "fast",
        channels_override: Optional[int] = None,
        solver_override: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.variant = variant
        
        if algebra_config is None:
            from attractor.models.clifford_attractor import CliffordAttractorConfig
            # The benchmark uses a small set of explicit Clifford variants so we can
            # isolate whether quality is driven by the sequence mixer or the solver.
            if variant not in {"fast", "no_mixer", "deep_solve"}:
                raise ValueError(f"Unknown CliffordAttractorLM variant: {variant}")
            if solver_override is not None and solver_override not in {"light", "default", "deep"}:
                raise ValueError(f"Unknown CliffordAttractorLM solver_override: {solver_override}")

            channels = channels_override if channels_override is not None else max(16, config.n_embd // 8)
            hidden_channels = channels * 2
            max_iter = 4
            tol = 2e-3
            anderson_m = 1
            use_sequence_mixer = True

            if variant == "no_mixer":
                use_sequence_mixer = False
            elif variant == "deep_solve":
                max_iter = 8
                tol = 1e-3
                anderson_m = 2

            if solver_override == "light":
                max_iter = 2
                tol = 3e-3
                anderson_m = 0
            elif solver_override == "default":
                max_iter = 4
                tol = 2e-3
                anderson_m = 1
            elif solver_override == "deep":
                max_iter = 8
                tol = 1e-3
                anderson_m = 2

            algebra_config = CliffordAttractorConfig(
                p=3, q=0, r=0,
                channels=channels,
                hidden_channels=hidden_channels,
                num_blocks=2,
                num_rotors=2,
                max_iter=max_iter,
                tol=tol,
                anderson_m=anderson_m,
                max_seq_len=config.block_size,
                use_sequence_mixer=use_sequence_mixer,
            )
        
        from attractor.models.clifford_attractor import CliffordAttractor
        self.attractor = CliffordAttractor(algebra_config, vocab_size=config.vocab_size)
        
    def forward(self, idx, targets=None, pad_token_id=0):
        logits = self.attractor(idx)
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=pad_token_id
            )
            return logits, loss
        return logits, None


class CausalSequenceProxyBaseline(nn.Module):
    """Non-Clifford baseline with the same causal front-end as CliffordAttractorLM.

    This is a proxy for the missing original attractor baseline in this checkout.
    It keeps token/position embeddings and a causal GRU, but removes the Clifford
    fixed-point core.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        feature_dim = (config.n_embd // 4) * 8
        self.token_embed = nn.Embedding(config.vocab_size, feature_dim)
        self.pos_embed = nn.Embedding(config.block_size, feature_dim)
        self.sequence_mixer = nn.GRU(feature_dim, feature_dim, batch_first=True)
        self.sequence_gate = nn.Parameter(torch.tensor(0.5))
        self.output_gate = nn.Parameter(torch.tensor(-2.0))
        self.output_proj = nn.Linear(feature_dim, config.vocab_size)

    def forward(self, idx, targets=None, pad_token_id=0):
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size={self.config.block_size}")

        tok = self.token_embed(idx)
        pos = torch.arange(T, device=idx.device, dtype=torch.long)
        emb = tok + self.pos_embed(pos).unsqueeze(0)
        mixed, _ = self.sequence_mixer(emb)
        mix_gate = torch.sigmoid(self.sequence_gate)
        x = mix_gate * mixed + (1.0 - mix_gate) * emb
        output_mix = torch.sigmoid(self.output_gate)
        logits = self.output_proj(output_mix * x + (1.0 - output_mix) * emb)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=pad_token_id,
            )
            return logits, loss
        return logits, None


class MiniCliffordAttractor(nn.Module):
    def __init__(self, config: GPTConfig, algebra_dim: int = 16):
        super().__init__()
        self.config = config
        self.algebra_dim = algebra_dim

        self.input_proj = nn.Linear(config.n_embd, algebra_dim, bias=True)
        self.gp_kernel = nn.Parameter(torch.eye(algebra_dim).unsqueeze(0).repeat(algebra_dim, 1, 1))
        self.gp_bias = nn.Parameter(torch.zeros(algebra_dim))
        self.output_proj = nn.Linear(algebra_dim, config.n_embd, bias=True)
        self.ln = nn.LayerNorm(config.n_embd)

        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def geometric_product(self, x):
        # Simplified Clifford GP: element-wise product with learnable diagonal
        kernel = torch.tanh(self.gp_kernel.mean(dim=0))  # [algebra_dim, algebra_dim]
        out = torch.einsum('btc,cd->btd', x, kernel)
        return out + self.gp_bias

    def forward(self, idx, targets=None, pad_token_id=0):
        B, T = idx.size()

        x = self.embedding(idx)
        x_clifford = self.input_proj(x)

        for _ in range(4):
            gp_out = torch.tanh(self.geometric_product(x_clifford))
            x_clifford = 0.7 * x_clifford + 0.3 * gp_out

        x = self.output_proj(x_clifford)
        x = self.ln(x)

        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token_id)

        return logits, loss


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

        print('Tokenizing {} text samples...'.format(len(texts)))
        self.data = []
        for i, text in enumerate(texts):
            if len(text.strip()) < 10:
                continue
            try:
                encoding = tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=block_size,
                    padding=False
                )
                ids = encoding['input_ids']
                if len(ids) > 4:
                    self.data.append(torch.tensor(ids, dtype=torch.long))
            except Exception:
                continue

            if (i + 1) % 500 == 0:
                print('  Processed {}/{} texts, kept {} valid sequences'.format(i + 1, len(texts), len(self.data)))

        print('Dataset created with {} sequences'.format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][:-1]
        y = self.data[idx][1:]
        return x, y


def pad_collate(batch, pad_token_id=0):
    x, y = zip(*batch)
    lengths = [len(seq) for seq in x]
    max_len = max(lengths)

    x_padded = torch.full((len(x), max_len), pad_token_id, dtype=torch.long)
    y_padded = torch.full((len(y), max_len), pad_token_id, dtype=torch.long)

    for i, (seq_x, seq_y) in enumerate(zip(x, y)):
        x_padded[i, :len(seq_x)] = seq_x
        y_padded[i, :len(seq_y)] = seq_y

    return x_padded, y_padded


def load_sample_texts():
    from datasets import load_dataset

    print('Loading wikitext-2 dataset...')
    try:
        ds = load_dataset('wikitext', 'wikitext-2-v1', split='train', streaming=True)
        ds_iter = iter(ds)

        texts = []
        while len(texts) < 1000:
            try:
                item = next(ds_iter)
                text = item.get('text', '')
                if len(text.strip()) > 20:
                    texts.append(text)
            except StopIteration:
                break

        print('Loaded {} text samples'.format(len(texts)))
        return texts
    except Exception as e:
        print('Error loading dataset: {}'.format(e))
        return None


def train_epoch(model, dataloader, optimizer, device, max_batches=None):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    start_time = time.time()
    pad_token_id = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        _, loss = model(x, y, pad_token_id=pad_token_id)

        if loss is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if device.type == 'cuda':
                torch.cuda.synchronize()

            total_loss += loss.item()
            total_tokens += x.numel()
            num_batches += 1

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, num_batches)
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

    return avg_loss, tokens_per_sec, elapsed


def profile_train_step(model, dataloader, optimizer, device, num_batches: int = 3):
    """Profile forward/backward/step timing for a few batches."""
    model.train()
    pad_token_id = 0
    records = []
    iterator = iter(dataloader)

    for _ in range(num_batches):
        x, y = next(iterator)
        x, y = x.to(device), y.to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()

        optimizer.zero_grad()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()

        _, loss = model(x, y, pad_token_id=pad_token_id)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t2 = time.time()

        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t3 = time.time()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t4 = time.time()

        records.append({
            'zero_grad': t1 - t0,
            'forward_loss': t2 - t1,
            'backward': t3 - t2,
            'step': t4 - t3,
            'total': t4 - t0,
            'loss': float(loss.detach()),
            'tokens': int(x.numel()),
        })

    return records


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=20):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    pad_token_id = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y, pad_token_id=pad_token_id)
        if loss is not None:
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)


@dataclass
class BenchmarkConfig:
    name: str
    model_fn: type
    model_kwargs: dict
    batch_size: int = 16
    block_size: int = 128
    lr: float = 1e-3
    epochs: int = 20
    max_train_batches: int = 100
    max_eval_batches: int = 30
    profile_batches: int = 0


def run_benchmark(config: BenchmarkConfig, tokenizer, texts, device):
    print('=' * 70)
    print('Benchmark: {}'.format(config.name))
    print('=' * 70)

    print('Creating model...')
    model = config.model_fn(**config.model_kwargs)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print('Model parameters: {:,}'.format(num_params))

    print('Creating dataset...')
    dataset = TextDataset(texts, tokenizer, config.block_size)
    if len(dataset) == 0:
        print('ERROR: Dataset is empty!')
        return None

    eval_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - eval_size
    if train_size <= 0:
        train_size = 1
        eval_size = len(dataset) - train_size
    split_generator = torch.Generator().manual_seed(SEED)
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size], generator=split_generator)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=0, collate_fn=pad_collate)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=0, collate_fn=pad_collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

    if config.profile_batches > 0:
        print('Profiling {} train steps...'.format(config.profile_batches))
        profile_records = profile_train_step(
            model, train_loader, optimizer, device, num_batches=config.profile_batches
        )
        for i, rec in enumerate(profile_records, 1):
            print(
                '  Profile batch {}/{}: loss={:.4f}, total={:.3f}s, zero_grad={:.3f}s, forward={:.3f}s, backward={:.3f}s, step={:.3f}s'.format(
                    i, config.profile_batches, rec['loss'], rec['total'], rec['zero_grad'], rec['forward_loss'], rec['backward'], rec['step']
                )
            )

    results = {
        'name': config.name,
        'params': num_params,
        'train_losses': [],
        'eval_losses': [],
        'times': [],
        'profile': profile_records if config.profile_batches > 0 else [],
    }

    total_start = time.time()

    for epoch in range(config.epochs):
        epoch_start = time.time()

        train_loss, tok_per_sec, _ = train_epoch(
            model, train_loader, optimizer, device,
            max_batches=config.max_train_batches
        )

        eval_loss = evaluate(model, eval_loader, device, max_batches=config.max_eval_batches)

        epoch_time = time.time() - epoch_start

        results['train_losses'].append(train_loss)
        results['eval_losses'].append(eval_loss)
        results['times'].append(epoch_time)

        print('  Epoch {}/{}: train_loss={:.4f}, eval_loss={:.4f}, time={:.1f}s, tok/s={:.0f}'.format(
            epoch + 1, config.epochs, train_loss, eval_loss, epoch_time, tok_per_sec))

    total_time = time.time() - total_start

    results['total_time'] = total_time
    results['avg_epoch_time'] = sum(results['times']) / len(results['times'])
    results['final_train_loss'] = results['train_losses'][-1]
    results['final_eval_loss'] = results['eval_losses'][-1]
    results['throughput'] = tok_per_sec

    print('Summary for {}:'.format(config.name))
    print('  Parameters: {:,}'.format(num_params))
    print('  Total time: {:.1f}s'.format(total_time))
    print('  Final train loss: {:.4f}'.format(results['final_train_loss']))
    print('  Final eval loss: {:.4f}'.format(results['final_eval_loss']))

    return results


def main():
    from transformers import AutoTokenizer

    print('Loading tokenizer (GPT2)...')
    try:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        print('Tokenizer vocab size: {}'.format(len(tokenizer)))
    except Exception as e:
        print('Error loading tokenizer: {}'.format(e))
        return

    texts = load_sample_texts()
    if texts is None or len(texts) == 0:
        print('ERROR: Could not load text data')
        return

    benchmarks = [
        BenchmarkConfig(
            name='RepoNativeGPT-Small',
            model_fn=RepoNativeGPTBaseline,
            model_kwargs={'config': GPTConfig(
                vocab_size=len(tokenizer),
                block_size=128,
                n_embd=256,
                n_layer=4,
                n_head=4
            )},
            batch_size=16,
            lr=1e-3,
            epochs=20,
            max_train_batches=100
        ),
        BenchmarkConfig(
            name='RepoNativeGPT-Medium',
            model_fn=RepoNativeGPTBaseline,
            model_kwargs={'config': GPTConfig(
                vocab_size=len(tokenizer),
                block_size=128,
                n_embd=384,
                n_layer=6,
                n_head=6
            )},
            batch_size=16,
            lr=1e-3,
            epochs=20,
            max_train_batches=100
        ),
        BenchmarkConfig(
            name='MiniCliffordAttractor',
            model_fn=MiniCliffordAttractor,
            model_kwargs={'config': GPTConfig(
                vocab_size=len(tokenizer),
                block_size=128,
                n_embd=256,
                n_layer=4,
                n_head=4
            ), 'algebra_dim': 16},
            batch_size=16,
            lr=1e-3,
            epochs=20,
            max_train_batches=100
        ),
        BenchmarkConfig(
            name='CausalSequenceProxyBaseline',
            model_fn=CausalSequenceProxyBaseline,
            model_kwargs={'config': GPTConfig(
                vocab_size=len(tokenizer),
                block_size=128,
                n_embd=256,
                n_layer=4,
                n_head=4
            )},
            batch_size=16,
            lr=1e-3,
            epochs=20,
            max_train_batches=100
        ),
        BenchmarkConfig(
            name='CliffordAttractor-LM-24ch-LightSolve',
            model_fn=CliffordAttractorLM,
            model_kwargs={'config': GPTConfig(
                vocab_size=len(tokenizer),
                block_size=128,
                n_embd=256,
                n_layer=4,
                n_head=4
            ), 'channels_override': 24, 'solver_override': 'light'},
            batch_size=16,
            lr=1e-3,
            epochs=20,
            max_train_batches=100,
            profile_batches=3
        ),
        BenchmarkConfig(
            name='CliffordAttractor-LM-24ch',
            model_fn=CliffordAttractorLM,
            model_kwargs={'config': GPTConfig(
                vocab_size=len(tokenizer),
                block_size=128,
                n_embd=256,
                n_layer=4,
                n_head=4
            ), 'channels_override': 24},
            batch_size=16,
            lr=1e-3,
            epochs=20,
            max_train_batches=100
        ),
        BenchmarkConfig(
            name='CliffordAttractor-LM-24ch-DeepSolve',
            model_fn=CliffordAttractorLM,
            model_kwargs={'config': GPTConfig(
                vocab_size=len(tokenizer),
                block_size=128,
                n_embd=256,
                n_layer=4,
                n_head=4
            ), 'channels_override': 24, 'solver_override': 'deep'},
            batch_size=16,
            lr=1e-3,
            epochs=20,
            max_train_batches=100
        ),
        BenchmarkConfig(
            name='CliffordAttractor-LM-40ch',
            model_fn=CliffordAttractorLM,
            model_kwargs={'config': GPTConfig(
                vocab_size=len(tokenizer),
                block_size=128,
                n_embd=256,
                n_layer=4,
                n_head=4
            ), 'channels_override': 40},
            batch_size=16,
            lr=1e-3,
            epochs=20,
            max_train_batches=100
        ),
        BenchmarkConfig(
            name='CliffordAttractor-LM-NoMixer',
            model_fn=CliffordAttractorLM,
            model_kwargs={'config': GPTConfig(
                vocab_size=len(tokenizer),
                block_size=128,
                n_embd=256,
                n_layer=4,
                n_head=4
            ), 'variant': 'no_mixer'},
            batch_size=16,
            lr=1e-3,
            epochs=20,
            max_train_batches=100
        ),
        BenchmarkConfig(
            name='CliffordAttractor-LM-DeepSolve',
            model_fn=CliffordAttractorLM,
            model_kwargs={'config': GPTConfig(
                vocab_size=len(tokenizer),
                block_size=128,
                n_embd=256,
                n_layer=4,
                n_head=4
            ), 'variant': 'deep_solve'},
            batch_size=16,
            lr=1e-3,
            epochs=20,
            max_train_batches=100
        ),
    ]

    all_results = []

    for bench_cfg in benchmarks:
        result = run_benchmark(bench_cfg, tokenizer, texts, DEVICE)
        if result:
            all_results.append(result)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary table
    print('=' * 70)
    print('BENCHMARK SUMMARY')
    print('=' * 70)
    header = '{:<25} {:>12} {:>10} {:>12} {:>12} {:>10}'.format(
        'Model', 'Params', 'Time(s)', 'Train Loss', 'Eval Loss', 'Tok/s')
    print(header)
    print('-' * 80)

    for r in all_results:
        row = '{:<25} {:>10,} {:>9.1f} {:>11.4f} {:>11.4f} {:>9.0f}'.format(
            r['name'], r['params'], r['total_time'],
            r['final_train_loss'], r['final_eval_loss'], r['throughput'])
        print(row)

    output_path = 'experiments/benchmark_lm_results.pt'
    torch.save(all_results, output_path)
    print('Results saved to {}'.format(output_path))

    # Plot loss curves
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Training loss curves
        ax1 = axes[0]
        for r in all_results:
            epochs = range(1, len(r['train_losses']) + 1)
            ax1.plot(epochs, r['train_losses'], marker='o', label=r['name'], linewidth=2, markersize=4)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Training Loss', fontsize=12)
        ax1.set_title('Training Loss Curves', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Evaluation loss curves
        ax2 = axes[1]
        for r in all_results:
            epochs = range(1, len(r['eval_losses']) + 1)
            ax2.plot(epochs, r['eval_losses'], marker='s', label=r['name'], linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Evaluation Loss', fontsize=12)
        ax2.set_title('Evaluation Loss Curves', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = 'experiments/benchmark_lm_loss_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print('Loss curves saved to {}'.format(plot_path))

    except Exception as e:
        print('Warning: Could not generate plots: {}'.format(e))


if __name__ == '__main__':
    main()
