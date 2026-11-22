"""Behavior checks for the event-order training and generation workflow.

All records and vectors here are synthetic. No PopHR file is read by these tests.
"""

import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from data.data_pipeline import EHRDataset
from models.TrajGPT import TrajGPT


VOCABULARY = [11, 22, 33, 44]


def make_config():
    return SimpleNamespace(
        num_layers=1,
        num_heads=4,
        d_model=200,
        qk_dim=40,
        v_dim=40,
        dropout=0.0,
        tau=20.0,
        ffn_proj_size=64,
        use_grad_ckp=False,
        use_bias_in_sra=False,
        use_bias_in_mlp=True,
        use_bias_in_sra_out=False,
        use_default_gamma=False,
        forward_impl="parallel",
        seq_len=256,
    )


def make_model(device="cpu"):
    generator = torch.Generator().manual_seed(87)
    model = TrajGPT(
        make_config(),
        embedding_data={
            "phecodes": VOCABULARY,
            "embeddings": torch.randn(4, 200, generator=generator),
        },
    )
    return model.to(device)


def records(codes):
    return pd.DataFrame(
        {
            "id": ["synthetic_patient"] * len(codes),
            "date": pd.date_range("2000-01-01", periods=len(codes)).astype(str),
            "PheCode": codes,
            "age_at_diag": [100 + i for i in range(len(codes))],
            "month_of_birth": ["1990-01"] * len(codes),
        }
    )


@pytest.mark.parametrize("length", [3, 4, 6])
def test_latest_events_are_selected_before_teacher_forcing(length):
    source_codes = [11, 22, 33, 44, 11, 22][:length]
    frame = records(source_codes)
    original_frame = frame.copy(deep=True)
    dataset = EHRDataset(frame, max_length=4)
    example = dataset[0]
    kept_codes = source_codes[-4:]
    kept_times = [100 + i for i in range(length)][-4:]
    count = len(kept_codes)

    assert example["input_ids"][:count].tolist() == [-1] + kept_codes[:-1]
    assert example["labels"][:count].tolist() == kept_codes
    assert example["input_time"][:count].tolist() == [0] + kept_times[:-1]
    assert example["labels"][count - 1].item() == source_codes[-1]
    assert example["attention_mask"].tolist() == [True] * count + [False] * (4 - count)
    assert example["input_ids"][count:].eq(0).all()
    assert example["labels"][count:].eq(0).all()
    assert example["input_ids"].dtype == torch.long
    assert example["attention_mask"].dtype == torch.bool
    pd.testing.assert_frame_equal(frame, original_frame)
    # Re-reading a patient must not repeatedly shift or mutate its stored list.
    assert torch.equal(dataset[0]["input_ids"], example["input_ids"])


def test_single_event_keeps_a_valid_sos_and_a_real_target():
    example = EHRDataset(records([44]), max_length=4)[0]
    assert example["input_ids"].tolist() == [-1, 0, 0, 0]
    assert example["labels"].tolist() == [44, 0, 0, 0]
    assert example["attention_mask"].tolist() == [True, False, False, False]


def test_same_day_events_keep_their_original_order():
    frame = records([33, 11, 44, 22])
    frame["date"] = "2000-01-01"
    example = EHRDataset(frame, max_length=4)[0]
    assert example["labels"].tolist() == [33, 11, 44, 22]


def test_model_does_not_shift_inputs_a_second_time():
    model = make_model().eval()
    inputs = torch.tensor([[-1, 11, 22, 44]])
    seen = []

    def capture_input(_module, args):
        seen.append(args[0].detach().clone())

    hook = model.SRA_blocks[0].register_forward_pre_hook(capture_input)
    try:
        with torch.no_grad():
            output = model(inputs)
    finally:
        hook.remove()

    assert output["loss"] is None
    assert output["logits"].shape == (1, 4, 5)
    torch.testing.assert_close(seen[0][0, 0], model.sos)
    torch.testing.assert_close(seen[0][0, 1:], model.token_embedding.emb.weight[[0, 1, 3]])


def test_future_events_cannot_change_earlier_predictions():
    model = make_model().eval()
    original = torch.tensor([[-1, 11, 22, 33, 44]])
    changed = torch.tensor([[-1, 11, 22, 44, 11]])
    with torch.no_grad():
        before = model(original)["logits"]
        after = model(changed)["logits"]
    torch.testing.assert_close(before[:, :3], after[:, :3], rtol=1e-5, atol=1e-6)


def test_old_unshifted_training_call_fails_instead_of_learning_the_answer():
    model = make_model()
    old_inputs = torch.tensor([[11, 22, 33]])
    with pytest.raises(ValueError, match='SOS'):
        model(old_inputs, y=old_inputs)


def test_right_padding_preserves_valid_logits_and_token_mean_loss():
    model = make_model().eval()
    short_inputs = torch.tensor([[-1, 11, 22]])
    short_targets = torch.tensor([[11, 22, 33]])
    padded_inputs = torch.tensor([[-1, 11, 22, 0, 0, 0]])
    padded_targets = torch.tensor([[11, 22, 33, 0, 0, 0]])
    with torch.no_grad():
        short = model(short_inputs, y=short_targets, retention_mask=short_inputs.ne(0))
        padded = model(padded_inputs, y=padded_targets, retention_mask=padded_inputs.ne(0))
    torch.testing.assert_close(short["logits"], padded["logits"][:, :3], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(short["loss"], padded["loss"], rtol=1e-5, atol=1e-6)
    # Independent target-index calculation checks the raw-code vocabulary mapping.
    expected_loss = F.cross_entropy(short["logits"].reshape(-1, 5), torch.tensor([0, 1, 2]))
    torch.testing.assert_close(short["loss"], expected_loss)


def test_training_updates_weights_but_not_fixed_embeddings():
    model = make_model().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    embeddings_before = model.token_embedding.emb.weight.detach().clone()
    head_before = model.head.head.weight.detach().clone()
    inputs = torch.tensor([[-1, 11, 22, 0], [-1, 33, 44, 11]])
    labels = torch.tensor([[11, 22, 33, 0], [33, 44, 11, 22]])
    result = model(inputs, y=labels, retention_mask=inputs.ne(0))
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(g).all() for g in gradients)
    assert model.head.head.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(head_before, model.head.head.weight)
    assert not model.token_embedding.emb.weight.requires_grad
    assert model.token_embedding.emb.weight.grad is None
    torch.testing.assert_close(embeddings_before, model.token_embedding.emb.weight, rtol=0, atol=0)


class RecordingModel(torch.nn.Module):
    """A deterministic oracle makes feedback and PAD suppression observable."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.token_embedding = SimpleNamespace(phecodes=VOCABULARY, padding_idx=4)
        self.seen_inputs = []

    def forward(self, X, input_time=None, y=None, **kwargs):
        assert y is None, "Generation must not receive future ground-truth labels."
        self.seen_inputs.append(X.detach().cpu().tolist()[0])
        assert X[0, 0].item() == -1
        assert X.ne(0).all(), "Single-history generation should not append PAD to find the next token."
        logits = torch.full((*X.shape, 5), -10.0, device=X.device)
        logits[..., 4] = 100.0  # Without explicit masking, greedy generation emits PAD.
        previous_code = int(X[0, -1].item())
        next_index = (VOCABULARY.index(previous_code) + 1) % len(VOCABULARY)
        logits[0, -1, next_index] = 10.0
        return {"logits": logits, "loss": None}


def test_generation_uses_latest_event_and_feeds_back_predictions():
    from run_generate import generate_codes

    model = RecordingModel()
    history = [11, 22, 33]
    generated = generate_codes(model, history, max_new_tokens=3, context_length=4)
    assert generated == [44, 11, 22]
    assert model.seen_inputs == [
        [-1, 11, 22, 33],
        [-1, 22, 33, 44],
        [-1, 33, 44, 11],
    ]
    assert history == [11, 22, 33]
    assert all(code in VOCABULARY for code in generated)


def test_generation_window_removes_oldest_events():
    from run_generate import build_generation_input

    inputs = build_generation_input([11, 22, 33, 44], context_length=4, device=torch.device("cpu"))
    assert inputs.tolist() == [[-1, 22, 33, 44]]


def test_checkpoint_restores_predictions_and_optimizer_state(tmp_path):
    from training_utils import load_checkpoint, save_checkpoint

    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.tensor([[-1, 11, 22, 33]])
    labels = torch.tensor([[11, 22, 33, 44]])

    def update(target_model, target_optimizer):
        target_model.train()
        target_optimizer.zero_grad()
        target_model(inputs, y=labels)["loss"].backward()
        target_optimizer.step()

    update(model, optimizer)
    model.eval()
    with torch.no_grad():
        expected = model(inputs)["logits"].clone()
    split_ids = {"train": ["synthetic_train"], "valid": ["synthetic_valid"], "test": ["synthetic_test"]}
    checkpoint = tmp_path / "roundtrip.pt"
    save_checkpoint(
        checkpoint, model, optimizer, vars(make_config()), split_ids,
        epoch=1, global_step=7, best_val_loss=1.25,
    )
    loaded, payload = load_checkpoint(checkpoint, device="cpu")
    loaded.eval()
    with torch.no_grad():
        actual = loaded(inputs)["logits"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert payload["epoch"] == 1
    assert payload["global_step"] == 7
    assert payload["best_val_loss"] == 1.25
    assert payload["split_ids"] == split_ids
    assert loaded.token_embedding.phecodes == VOCABULARY
    restored_optimizer = torch.optim.Adam(loaded.parameters(), lr=0.9)
    restored_optimizer.load_state_dict(payload["optimizer_state_dict"])
    assert restored_optimizer.param_groups[0]["lr"] == 1e-3
    # A second identical update checks optimizer moments, not merely model tensors.
    update(model, optimizer)
    update(loaded, restored_optimizer)
    for left, right in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")),
    ],
)
def test_forward_and_backward_on_available_devices(device):
    model = make_model(device)
    inputs = torch.tensor([[-1, 11, 22]], device=device)
    labels = torch.tensor([[11, 22, 33]], device=device)
    result = model(inputs, y=labels)
    assert result["logits"].device.type == device
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


class TokenLossModel(torch.nn.Module):
    """Tiny independent objective for checking loss weighting and accumulation."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.3))

    def forward(self, X, input_time=None, y=None, **kwargs):
        valid = y.ne(0)
        scores = self.weight * X[valid].float()
        labels = y[valid].eq(1).float()
        return {"loss": F.binary_cross_entropy_with_logits(scores, labels)}


def variable_length_batches():
    batches = []
    for inputs, labels in [
        ([3, 0, 0, 0], [1, 0, 0, 0]),
        ([1, 2, 3, 0], [1, 2, 1, 0]),
        ([2, 4, 0, 0], [2, 2, 0, 0]),
    ]:
        input_ids = torch.tensor([inputs])
        targets = torch.tensor([labels])
        batches.append({
            "input_ids": input_ids,
            "input_time": torch.zeros_like(input_ids, dtype=torch.float32),
            "labels": targets,
            "attention_mask": targets.ne(0),
        })
    return batches


def independent_token_loss(model, batches):
    values = torch.cat([batch["input_ids"][batch["labels"].ne(0)] for batch in batches]).float()
    labels = torch.cat([batch["labels"][batch["labels"].ne(0)] for batch in batches]).eq(1).float()
    return F.binary_cross_entropy_with_logits(model.weight * values, labels)


def test_epoch_loss_is_weighted_by_valid_tokens_and_respects_batch_limit():
    from run_pretrain import run_epoch

    model = TokenLossModel()
    batches = variable_length_batches()
    result = run_epoch(model, batches, torch.device("cpu"))
    expected = independent_token_loss(model, batches).item()
    assert result == pytest.approx({"loss": expected, "tokens": 6, "batches": 3, "updates": 0})
    unweighted = sum(independent_token_loss(model, [batch]).item() for batch in batches) / 3
    assert abs(expected - unweighted) > 0.01

    limited = run_epoch(model, batches, torch.device("cpu"), max_batches=2)
    assert limited["tokens"] == 4
    assert limited["batches"] == 2
    assert limited["loss"] == pytest.approx(independent_token_loss(model, batches[:2]).item())


def test_accumulation_matches_token_weighted_updates_including_partial_group():
    from run_pretrain import run_epoch

    accumulated = TokenLossModel()
    reference = TokenLossModel()
    batches = variable_length_batches()
    optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    expected_nll = 0.0
    # Independently combine tokens: first update has four, final partial group has two.
    for group in [batches[:2], batches[2:]]:
        reference_optimizer.zero_grad()
        loss = independent_token_loss(reference, group)
        tokens = sum(int(batch["labels"].ne(0).sum()) for batch in group)
        expected_nll += loss.item() * tokens
        loss.backward()
        reference_optimizer.step()

    actual = run_epoch(
        accumulated, batches, torch.device("cpu"), optimizer=optimizer,
        accum_steps=2, grad_clip=0.0,
    )
    assert actual["updates"] == 2
    assert actual["tokens"] == 6
    assert actual["batches"] == 3
    assert actual["loss"] == pytest.approx(expected_nll / 6)
    torch.testing.assert_close(accumulated.weight, reference.weight, rtol=1e-6, atol=1e-7)


def test_resume_preserves_settings_progress_and_original_files(tmp_path, monkeypatch):
    import run_pretrain
    from training_utils import load_checkpoint, save_checkpoint

    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    initial_inputs = torch.tensor([[-1, 11, 22]])
    initial_labels = torch.tensor([[11, 22, 33]])
    model(initial_inputs, y=initial_labels)["loss"].backward()
    optimizer.step()
    optimizer.zero_grad()
    split_ids = {"train": ["t0", "t1", "t2"], "valid": ["v0"], "test": ["s0"]}
    config = vars(make_config()) | {
        "batch_size": 1,
        "learning_rate": 2e-4,
        "train_epochs": 1,
        "use_grad_accum": True,
        "accum_steps": 2,
        "grad_clip": 0.7,
        "patience": 3,
        "max_train_batches": 3,
        "max_valid_batches": 1,
    }
    source_dir = tmp_path / "original"
    source_dir.mkdir()
    source_path = source_dir / "last.pt"
    marker = source_dir / "keep.txt"
    marker.write_text("Existing run must remain unchanged.\n")
    # A best value of zero makes this next validation a non-improvement. The
    # restored bad-epoch count therefore triggers early stopping after epoch 2.
    save_checkpoint(
        source_path, model, optimizer, config, split_ids,
        epoch=1, global_step=1, best_val_loss=0.0, extra={"bad_epochs": 2},
    )
    original_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_dir.iterdir()}
    provider_calls = []

    def synthetic_provider(batch_size, max_length, split_ids=None, return_split_ids=False):
        provider_calls.append((batch_size, max_length, split_ids, return_split_ids))
        assert batch_size == 1
        assert max_length == 256
        assert return_split_ids is True
        assert split_ids == {"train": ["t0", "t1", "t2"], "valid": ["v0"], "test": ["s0"]}
        datasets = []
        loaders = []
        for split in ("train", "valid", "test"):
            frames = []
            for patient_id in split_ids[split]:
                frame = records([11, 22, 33])
                frame["id"] = patient_id
                frames.append(frame)
            # Tiny synthetic contexts keep this control-flow test inexpensive.
            dataset = EHRDataset(pd.concat(frames, ignore_index=True), max_length=4)
            datasets.append(dataset)
            loaders.append(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False))
        return datasets[0], loaders[0], datasets[1], loaders[1], datasets[2], loaders[2], split_ids

    monkeypatch.setattr(run_pretrain, "data_provider", synthetic_provider)
    arguments = [
        "--resume", str(source_path), "--train_epochs", "3",
        "--device", "cpu", "--cpu_threads", "1",
        "--checkpoints", str(tmp_path), "--run_name", "resumed",
    ]
    result_dir = run_pretrain.main(arguments)
    assert result_dir == tmp_path / "resumed"
    assert len(provider_calls) == 1
    saved_config = json.loads((result_dir / "config.json").read_text())
    for key in ("batch_size", "learning_rate", "use_grad_accum", "accum_steps", "grad_clip",
                "patience", "max_train_batches", "max_valid_batches"):
        assert saved_config[key] == config[key]
    assert saved_config["train_epochs"] == 3
    history = json.loads((result_dir / "history.json").read_text())
    assert [row["epoch"] for row in history] == [2]
    assert history[0]["train_batches"] == 3
    assert history[0]["valid_batches"] == 1
    assert history[0]["global_step"] == 3  # Previous update plus two accumulated updates.
    _, final_payload = load_checkpoint(result_dir / "last.pt")
    _, initial_payload = load_checkpoint(result_dir / "initial.pt")
    assert initial_payload["epoch"] == 1
    assert initial_payload["global_step"] == 1
    assert final_payload["epoch"] == 2
    assert final_payload["global_step"] == 3
    assert final_payload["extra"]["bad_epochs"] == 3
    assert final_payload["split_ids"] == split_ids
    assert all(int(state["step"]) == 3 for state in final_payload["optimizer_state_dict"]["state"].values())
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_dir.iterdir()} == original_files

    # A repeated explicit run name must fail before changing any completed files.
    resumed_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in result_dir.iterdir()}
    with pytest.raises(FileExistsError):
        run_pretrain.main(arguments)
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in result_dir.iterdir()} == resumed_files
