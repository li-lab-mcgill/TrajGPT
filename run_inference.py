"""TrajGPT autoregressive inference at given future event times, with evaluation."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from data.data_pipeline import DATA_ROW_LIMIT, SOS_TOKEN, data_provider
from training_utils import SOURCE_ROOT, load_checkpoint, select_device


def build_generation_input(history, context_length, device):
    """Compatibility helper for earlier event-order checkpoints."""
    if context_length < 2:
        raise ValueError('context_length must allow SOS and at least one event')
    recent = list(history)[-(context_length - 1):]
    return torch.tensor([[SOS_TOKEN] + recent], dtype=torch.long, device=device)


@torch.inference_mode()
def generate_codes(model, history, max_new_tokens=20, context_length=256):
    """Compatibility path for event-order models; temporal models use inference()."""
    if getattr(model, 'time_mode', 'event_order') != 'event_order':
        raise ValueError('Use model.inference() with history_times and future_times for temporal checkpoints')
    if max_new_tokens < 0:
        raise ValueError('max_new_tokens cannot be negative')
    vocabulary = list(model.token_embedding.phecodes)
    valid_codes = set(vocabulary)
    history = list(history)
    if not history or any(code not in valid_codes for code in history):
        raise ValueError('history must contain known, non-padding PheCodes')
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    generated = []
    try:
        for _ in range(max_new_tokens):
            inputs = build_generation_input(history, context_length, device)
            result = model(X=inputs, retention_mask=torch.ones_like(inputs, dtype=torch.bool))
            scores = result['logits'][0, -1, :len(vocabulary)]
            if not torch.isfinite(scores).all().item():
                raise RuntimeError('Non-finite generation logits')
            code = vocabulary[int(scores.argmax().item())]
            generated.append(code)
            history.append(code)
            history = history[-(context_length - 1):]
    finally:
        model.train(was_training)
    return generated


def phenotype_names():
    path = SOURCE_ROOT / 'data' / 'phecode_definitions1.2.csv'
    names = {}
    if path.exists():
        with path.open(newline='') as source:
            for row in csv.DictReader(source):
                try:
                    code = float(row['phecode'])
                except (TypeError, ValueError):
                    continue
                if code.is_integer():
                    names[int(code)] = row['phenotype']
    return names


def parse_history(text):
    codes = []
    for item in text.split(','):
        number = float(item.strip())
        if not math.isfinite(number) or not number.is_integer() or number <= 0:
            raise ValueError('History must be comma-separated positive integer PheCodes')
        codes.append(int(number))
    return codes


def parse_times(text):
    return [float(item.strip()) for item in text.split(',')]


def evaluate_predictions(predictions, true_codes):
    """Score an already generated trajectory; this function never calls the model."""
    generated = predictions['phecodes']
    candidates = predictions['top_k_phecodes']
    truth = list(true_codes)
    if len(generated) != len(truth) or len(candidates) != len(truth):
        raise ValueError('Predictions, candidates and true codes must have equal lengths')
    hits = [predicted == actual for predicted, actual in zip(generated, truth)]
    top_hits = [actual in row for actual, row in zip(truth, candidates)]
    count = len(truth)
    metrics = {
        'n_events': count, 'correct': sum(hits),
        'top1_accuracy': sum(hits) / count if count else None,
        'top_k': len(candidates[0]) if count else 0,
        'top_k_correct': sum(top_hits),
        'top_k_accuracy': sum(top_hits) / count if count else None,
        'cumulative_by_horizon': [],
    }
    for horizon in sorted({h for h in [1, 5, 10, 20, 50, count] if 0 < h <= count}):
        metrics['cumulative_by_horizon'].append({
            'horizon': horizon, 'n_events': horizon,
            'top1_accuracy': sum(hits[:horizon]) / horizon,
            'top_k_accuracy': sum(top_hits[:horizon]) / horizon,
        })
    return metrics


def evaluate_patient(model, events, times, prefix_length=50, context_length=None,
                     top_k=5, max_new_tokens=None):
    """Keep held-out PheCodes outside the model's autoregressive interface."""
    events, times = list(events), list(times)
    if len(events) != len(times):
        raise ValueError('Every event must have its own timestamp')
    if type(prefix_length) is not int or not 1 <= prefix_length < len(events):
        raise ValueError('prefix_length must leave at least one unseen event')
    if max_new_tokens is not None and (type(max_new_tokens) is not int or max_new_tokens < 1):
        raise ValueError('max_new_tokens must be positive when supplied')
    stop = len(events) if max_new_tokens is None else min(len(events), prefix_length + max_new_tokens)
    history_codes, history_times = events[:prefix_length], times[:prefix_length]
    future_times = times[prefix_length:stop]
    predictions = model.inference(
        history_codes=history_codes, history_times=history_times,
        future_times=future_times, context_length=context_length, top_k=top_k,
    )
    # Future labels are accessed only after the complete rollout has returned.
    true_codes = events[prefix_length:stop]
    return {
        'history_phecodes': history_codes, 'history_times': history_times,
        'future_times': future_times, 'predictions': predictions,
        'true_phecodes': true_codes,
        'metrics': evaluate_predictions(predictions, true_codes),
    }


def select_test_patient(checkpoint, context_length, prefix_length, patient_index=None):
    """Select by history length only, using the checkpoint's saved test split."""
    *_, test_dataset, _test_loader = data_provider(
        batch_size=1, max_length=context_length, split_ids=checkpoint['split_ids'],
        time_mode=checkpoint['config'].get('time_mode', 'event_order'),
    )
    records = test_dataset.data
    lengths = [len(events) for events in records.PheCode]
    eligible = [index for index, length in enumerate(lengths) if length > prefix_length]
    if not eligible:
        raise ValueError(f'No test patient has more than {prefix_length} events')
    if patient_index is None:
        patient_index = max(eligible, key=lambda index: lengths[index])
        selection = 'longest_eligible_test_sequence_first_index_on_tie'
    else:
        if not 0 <= patient_index < len(records):
            raise ValueError('patient_index is outside the saved test split')
        if patient_index not in eligible:
            raise ValueError('Selected test patient must have more events than prefix_length')
        selection = 'explicit_test_patient_index'
    patient = records.iloc[patient_index]
    # Use chronological records, not shifted/truncated __getitem__ training input.
    return list(patient.PheCode), [float(t) for t in patient.age_at_diag], {
        'test_patient_index': patient_index, 'selection_rule': selection,
        'test_patients': len(records), 'eligible_test_patients': len(eligible),
        'total_patient_events': lengths[patient_index], 'data_rows_limit': DATA_ROW_LIMIT,
    }


def trajectory_rows(predictions, history_times, prefix_length, names, true_codes=None):
    rows = []
    previous_time = float(history_times[-1])
    for index, (code, time) in enumerate(zip(predictions['phecodes'], predictions['times'])):
        candidates = predictions['top_k_phecodes'][index]
        row = {
            'step': index + 1, 'event_index': prefix_length + index + 1,
            'query_time': time, 'delta_months': time - previous_time,
            'predicted_phecode': code, 'predicted_phenotype': names.get(code, 'Unknown phenotype'),
            'top_k_phecodes': candidates,
            'top_k_probabilities': predictions['top_k_probabilities'][index],
        }
        if true_codes is not None:
            actual = int(true_codes[index])
            row.update(true_phecode=actual, true_phenotype=names.get(actual, 'Unknown phenotype'),
                       correct=code == actual, top_k_correct=actual in candidates)
        rows.append(row)
        previous_time = time
    return rows


def save_results(output_path, result):
    """Write new JSON and companion CSV files, never replacing existing results."""
    output_path = Path(output_path)
    if output_path.suffix.lower() != '.json':
        raise ValueError('--output must have a .json extension')
    csv_path = output_path.with_suffix('.csv')
    for path in (output_path, csv_path):
        if path.exists():
            raise FileExistsError(path)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x') as output:
        output.write(text)
    rows = result['generated']
    if rows:
        with csv_path.open('x', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                                 for key, value in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--cpu_threads', type=int, default=4)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--history', help='Comma-separated PheCodes; otherwise select a test patient')
    source.add_argument('--patient_index', type=int, help='Index in sorted test patients, not a patient ID')
    parser.add_argument('--history_times', help='Observed event times in months, aligned with --history')
    parser.add_argument('--future_times', help='Given future query times in months, possibly repeated')
    parser.add_argument('--prefix_length', type=int, default=50)
    parser.add_argument('--max_new_tokens', type=int, default=None,
                        help='Optional cap; temporal default is all given future times')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--output', type=Path, help='New .json result and companion .csv; neither may exist')
    args = parser.parse_args(argv)
    if args.cpu_threads < 1 or args.prefix_length < 1 or args.top_k < 1:
        raise ValueError('cpu_threads, prefix_length and top_k must be positive')
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        raise ValueError('max_new_tokens must be positive')
    torch.set_num_threads(args.cpu_threads)
    device = select_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    context_length = int(checkpoint['config'].get('seq_len', 256))
    temporal = model.time_mode == 'time_conditioned'
    if args.history is None and (args.history_times is not None or args.future_times is not None):
        raise ValueError('Explicit time lists require --history; test evaluation uses recorded event times')
    if not temporal and (args.history_times is not None or args.future_times is not None):
        raise ValueError('An event-order checkpoint cannot use temporal query conditions')
    if temporal and args.history is not None and (args.history_times is None or args.future_times is None):
        raise ValueError('Temporal inference with --history requires --history_times and --future_times')
    selection, metrics, truth = {}, None, None
    if args.history is None:
        events, times, selection = select_test_patient(
            checkpoint, context_length, args.prefix_length, args.patient_index)
        history = events[:args.prefix_length]
        history_times = times[:args.prefix_length]
        source_kind = 'held_out_test_patient_prefix'
    else:
        history = parse_history(args.history)
        history_times = parse_times(args.history_times) if temporal else None
        source_kind = 'provided_history'
    if temporal:
        if args.history is None:
            evaluated = evaluate_patient(model, events, times, args.prefix_length,
                                         context_length, args.top_k, args.max_new_tokens)
            predictions, truth, metrics = (evaluated['predictions'], evaluated['true_phecodes'],
                                           evaluated['metrics'])
        else:
            future_times = parse_times(args.future_times)
            if args.max_new_tokens is not None:
                future_times = future_times[:args.max_new_tokens]
            predictions = model.inference(history, history_times, future_times,
                                          context_length=context_length, top_k=args.top_k)
        rows = trajectory_rows(predictions, history_times, len(history), phenotype_names(), truth)
    else:
        codes = generate_codes(model, history, args.max_new_tokens or 20, context_length)
        names = phenotype_names()
        rows = [{'step': step, 'predicted_phecode': code,
                 'predicted_phenotype': names.get(code, 'Unknown phenotype')}
                for step, code in enumerate(codes, 1)]
    result = {
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_epoch': checkpoint['epoch'],
        'device': str(device), 'time_mode': model.time_mode,
        'task': 'phecode_prediction_given_future_event_times' if temporal else 'event_order_generation',
        'decoding': 'greedy_full_history_recomputation', 'source': source_kind, **selection,
        'prefix_length': len(history), 'context_length_including_sos': context_length,
        'history_phecodes': [int(code) for code in history], 'generated': rows, 'metrics': metrics,
    }
    if temporal:
        result.update(
            time_unit=checkpoint['config']['time_unit'], time_scale=checkpoint['config']['time_scale'],
            history_times=history_times, future_times=predictions['times'],
            future_time_source='held_out_recorded_event_times' if truth is not None else 'provided_query_times',
            future_phecodes_used_as_model_inputs=False,
            time_is_predicted=False, probability_normalization='valid_phecodes_excluding_PAD',
        )
    if args.output:
        save_results(args.output, result)
        print(json.dumps({'device': str(device), 'time_mode': model.time_mode,
                          'test_patient_index': selection.get('test_patient_index'),
                          'prefix_length': len(history), 'generated_events': len(rows),
                          'metrics': metrics, 'output_json': str(args.output.resolve()),
                          'output_csv': str(args.output.with_suffix('.csv').resolve())},
                         indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
