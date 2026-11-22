"""Time-conditioned next-event pretraining on the existing 10,000-row PopHR slice."""
import argparse
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

from data.data_pipeline import DATA_PATH, DATA_ROW_LIMIT, data_provider
from models.TrajGPT import TrajGPT
from training_utils import (
    SOURCE_ROOT, TIME_CONFIG, load_checkpoint, restore_rng_state, save_checkpoint,
    seed_everything, select_device,
)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--random_seed', type=int, default=2021)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--train_epochs', type=int, default=None,
                        help='Total epoch number, including resumed epochs')
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--checkpoints', type=Path, default=SOURCE_ROOT / 'checkpoints')
    parser.add_argument('--run_name', default=None, help='New directory; existing directories are never reused')
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument('--time_mode', choices=['time_conditioned', 'event_order'], default=None,
                        help='New runs default to time_conditioned; resume preserves the saved mode')
    parser.add_argument('--time_scale', type=float, default=None,
                        help='Months per RoPE unit and target-gap unit (default 1); fixed on resume')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=200, choices=[200])
    parser.add_argument('--qk_dim', type=int, default=200)
    parser.add_argument('--v_dim', type=int, default=200)
    parser.add_argument('--ffn_proj_size', type=int, default=800)
    parser.add_argument('--tau', type=float, default=20.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_bias_in_sra', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use_bias_in_mlp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use_bias_in_sra_out', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use_grad_ckp', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use_grad_accum', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--accum_steps', type=int, default=None)
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None, help='0 disables early stopping')
    parser.add_argument('--max_train_batches', type=int, default=None, help='Smoke-run limit; 0 uses every batch')
    parser.add_argument('--max_valid_batches', type=int, default=None)
    parser.add_argument('--cpu_threads', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=256, choices=[256])
    parser.add_argument('--model', default='TrajGPT', choices=['TrajGPT'])
    parser.add_argument('--forward_impl', default='parallel', choices=['parallel'])
    return parser


def run_epoch(model, loader, device, optimizer=None, max_batches=0, accum_steps=1, grad_clip=1.0):
    """Mean over real target tokens, not an average of batch means."""
    training = optimizer is not None
    model.train(training)
    total_nll, total_tokens, updates, group_tokens = 0.0, 0, 0, 0
    batches = min(len(loader), max_batches) if max_batches else len(loader)
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(loader):
            if batch_index >= batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            time_kwargs = {}
            if getattr(model, 'time_mode', 'event_order') == 'time_conditioned':
                time_kwargs['target_time'] = batch['target_time']
            result = model(
                X=batch['input_ids'], input_time=batch['input_time'], y=batch['labels'],
                retention_mask=batch['attention_mask'], forward_impl='parallel',
                **time_kwargs,
            )
            loss = result['loss']
            tokens = int((batch['labels'] != 0).sum().item())
            if tokens == 0 or not torch.isfinite(loss).item():
                raise RuntimeError('Non-finite loss or no valid target tokens')
            total_nll += loss.detach().item() * tokens
            total_tokens += tokens
            if training:
                (loss * tokens).backward()
                group_tokens += tokens
                if (batch_index + 1) % accum_steps == 0 or batch_index + 1 == batches:
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(group_tokens)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip if grad_clip > 0 else float('inf'),
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    group_tokens = 0
    if not total_tokens:
        raise RuntimeError('The data loader produced no valid target tokens')
    return {'loss': total_nll / total_tokens, 'tokens': total_tokens,
            'batches': batches, 'updates': updates}


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.cpu_threads < 1:
        raise ValueError('cpu_threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    device = select_device(args.device)
    seed_everything(args.random_seed)
    payload = None
    if args.resume:
        model, payload = load_checkpoint(args.resume, device=device)
        config = dict(payload['config'])
    else:
        config = vars(args).copy()
        config['use_default_gamma'] = False
    if payload:
        saved_mode = config.get('time_mode', 'event_order')
        if args.time_mode is not None and args.time_mode != saved_mode:
            raise ValueError('Cannot change time_mode on resume; start a new pretraining run')
        if args.time_scale is not None and args.time_scale != config.get('time_scale', 1.0):
            raise ValueError('Cannot change time_scale on resume; start a new pretraining run')
        config['time_mode'] = saved_mode
        config['time_scale'] = config.get('time_scale', 1.0)
    else:
        config['time_mode'] = args.time_mode or 'time_conditioned'
        config['time_scale'] = args.time_scale if args.time_scale is not None else 1.0
    if not math.isfinite(config['time_scale']) or config['time_scale'] <= 0:
        raise ValueError('time_scale must be finite and positive')
    if config['time_mode'] == 'time_conditioned':
        config.update(TIME_CONFIG)
    for key, default in {'use_grad_accum': False, 'accum_steps': 4, 'grad_clip': 1.0,
                         'patience': 0, 'max_train_batches': 0, 'max_valid_batches': 0}.items():
        override = getattr(args, key)
        saved = config.get(key)
        setattr(args, key, override if override is not None else saved if saved is not None else default)
    if args.accum_steps < 1 or not math.isfinite(args.grad_clip) or args.grad_clip < 0:
        raise ValueError('accum_steps must be positive and grad_clip finite and nonnegative')
    if min(args.max_train_batches, args.max_valid_batches, args.patience) < 0:
        raise ValueError('Batch limits and patience must be nonnegative')
    config.update({
        'batch_size': args.batch_size if args.batch_size is not None else config.get('batch_size') or 32,
        'learning_rate': args.learning_rate if args.learning_rate is not None else config.get('learning_rate') or 3e-5,
        'train_epochs': args.train_epochs if args.train_epochs is not None else (payload['epoch'] + 1 if payload else 20),
        'device': args.device, 'selected_device': str(device), 'seq_len': 256,
        'use_grad_accum': args.use_grad_accum, 'accum_steps': args.accum_steps,
        'grad_clip': args.grad_clip, 'patience': args.patience,
        'max_train_batches': args.max_train_batches, 'max_valid_batches': args.max_valid_batches,
        'cpu_threads': args.cpu_threads, 'data_path': str(DATA_PATH), 'data_rows_limit': DATA_ROW_LIMIT,
        'resume': str(args.resume) if args.resume else None,
    })
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    if config['batch_size'] < 1 or config['learning_rate'] <= 0 or config['train_epochs'] < 1:
        raise ValueError('batch_size, learning_rate and train_epochs must be positive')
    if not math.isfinite(config['learning_rate']):
        raise ValueError('learning_rate must be finite')
    if payload is None:
        model = TrajGPT(SimpleNamespace(**config), head_type='pretrain').to(device)
    start_epoch = payload['epoch'] + 1 if payload else 1
    if config['train_epochs'] < start_epoch:
        raise ValueError(f'train_epochs must be at least {start_epoch} to continue this checkpoint')
    data_time_kwargs = {'time_mode': config['time_mode']} if config['time_mode'] == 'time_conditioned' else {}
    train_set, train_loader, valid_set, valid_loader, test_set, _, split_ids = data_provider(
        config['batch_size'], max_length=256,
        split_ids=payload['split_ids'] if payload else None, return_split_ids=True,
        **data_time_kwargs,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    if payload:
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        if args.learning_rate is not None:
            for group in optimizer.param_groups:
                group['lr'] = args.learning_rate
        restore_rng_state(payload['rng_state'])
    config['learning_rate'] = optimizer.param_groups[0]['lr']
    run_name = args.run_name or datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + config['time_mode'] + '-' + uuid.uuid4().hex[:6]
    if Path(run_name).name != run_name or run_name in ('.', '..'):
        raise ValueError('run_name must be one directory name')
    run_dir = args.checkpoints.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(message)s', force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / 'train.log')])
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (run_dir / 'split_ids.json').write_text(json.dumps(split_ids, indent=2) + '\n')
    logging.info('device=%s run_dir=%s', device, run_dir)
    logging.info('patients train=%d valid=%d test=%d; fixed data rows=%d; sequence length=256',
                 len(train_set), len(valid_set), len(test_set), DATA_ROW_LIMIT)
    logging.info('time_mode=%s; teacher forcing; latest events retained; pretraining only', config['time_mode'])
    if config['time_mode'] == 'time_conditioned':
        logging.info('real history time RoPE + target log1p-gap conditioning; unit=months scale=%s; SOS=first event time',
                     config['time_scale'])
    global_step = payload['global_step'] if payload else 0
    best_val_loss = payload['best_val_loss'] if payload else float('inf')
    bad_epochs = int(payload.get('extra', {}).get('bad_epochs', 0)) if payload else 0
    history = []
    if payload:
        save_checkpoint(run_dir / 'initial.pt', model, optimizer, config, split_ids,
                        payload['epoch'], global_step, best_val_loss, extra=payload.get('extra', {}))
    for epoch in range(start_epoch, config['train_epochs'] + 1):
        start_time = time.monotonic()
        train = run_epoch(model, train_loader, device, optimizer, args.max_train_batches,
                          args.accum_steps if args.use_grad_accum else 1, args.grad_clip)
        valid = run_epoch(model, valid_loader, device, max_batches=args.max_valid_batches)
        global_step += train['updates']
        improved = valid['loss'] < best_val_loss
        best_val_loss = min(best_val_loss, valid['loss'])
        bad_epochs = 0 if improved else bad_epochs + 1
        row = {'epoch': epoch, 'global_step': global_step, 'train_loss': train['loss'],
               'valid_loss': valid['loss'], 'train_tokens': train['tokens'],
               'valid_tokens': valid['tokens'], 'train_batches': train['batches'],
               'valid_batches': valid['batches'], 'seconds': time.monotonic() - start_time}
        history.append(row)
        (run_dir / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        logging.info('epoch=%d train_loss=%.6f valid_loss=%.6f updates=%d seconds=%.2f',
                     epoch, train['loss'], valid['loss'], global_step, row['seconds'])
        save_args = (model, optimizer, config, split_ids, epoch, global_step, best_val_loss)
        extra = {'metrics': row, 'bad_epochs': bad_epochs, 'best_in_this_run': improved}
        save_checkpoint(run_dir / 'last.pt', *save_args, extra=extra)
        if improved:
            save_checkpoint(run_dir / 'best.pt', *save_args, extra=extra)
        if args.patience and bad_epochs >= args.patience:
            logging.info('Early stopping after %d epochs without improvement', bad_epochs)
            break
    logging.info('Saved %s', run_dir / 'last.pt')
    return run_dir


if __name__ == '__main__':
    main()
