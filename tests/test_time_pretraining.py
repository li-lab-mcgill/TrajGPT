"""Synthetic checks for real-time history encoding and query-conditioned pretraining.

These tests do not run the generation script or inspect patient records.
"""
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from data.data_pipeline import EHRDataset
from models.TrajGPT import TrajGPT
from training_utils import load_checkpoint, save_checkpoint


CODES = [11, 22, 33, 44]


def make_config(**overrides):
    config = dict(
        num_layers=1, num_heads=4, d_model=200, qk_dim=40, v_dim=40,
        dropout=0.0, tau=20.0, ffn_proj_size=64, use_grad_ckp=False,
        use_bias_in_sra=False, use_bias_in_mlp=True,
        use_bias_in_sra_out=False, use_default_gamma=False,
        forward_impl="parallel", seq_len=256, time_mode="time_conditioned",
        time_unit="months", time_scale=1.0,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def make_model(**overrides):
    torch.manual_seed(202)
    generator = torch.Generator().manual_seed(71)
    return TrajGPT(
        make_config(**overrides),
        embedding_data={"phecodes": CODES,
                        "embeddings": torch.randn(4, 200, generator=generator)},
    )


def records(codes, times):
    return pd.DataFrame({
        "id": ["synthetic"] * len(codes),
        "date": pd.date_range("2000-01-01", periods=len(codes)).astype(str),
        "PheCode": codes, "age_at_diag": times,
        "month_of_birth": ["1990-01"] * len(codes),
    })


def example():
    return {
        "X": torch.tensor([[-1, 11, 22, 33]]),
        "input_time": torch.tensor([[100., 100., 103., 110.]]),
        "target_time": torch.tensor([[100., 103., 110., 112.]]),
        "y": torch.tensor([[11, 22, 33, 44]]),
        "retention_mask": torch.ones(1, 4, dtype=torch.bool),
    }


@pytest.mark.parametrize("length", [1, 3, 4, 6])
def test_newest_events_and_their_two_time_sequences_are_aligned(length):
    codes = [11, 22, 33, 44, 11, 22][:length]
    times = [100., 100., 102.5, 107., 120., 125.][:length]
    dataset = EHRDataset(records(codes, times), max_length=4,
                         time_mode="time_conditioned")
    item = dataset[0]
    kept_codes, kept_times = codes[-4:], times[-4:]
    count = len(kept_codes)
    assert item["input_ids"][:count].tolist() == [-1] + kept_codes[:-1]
    assert item["labels"][:count].tolist() == kept_codes
    assert item["input_time"][:count].tolist() == [kept_times[0]] + kept_times[:-1]
    assert item["target_time"][:count].tolist() == kept_times
    assert item["attention_mask"].tolist() == [True] * count + [False] * (4 - count)
    assert item["labels"][count - 1].item() == codes[-1]
    assert item["target_time"][count - 1].item() == times[-1]
    assert item["input_ids"][count:].eq(0).all()
    assert item["labels"][count:].eq(0).all()
    assert torch.equal(dataset[0]["input_time"], item["input_time"])


@pytest.mark.parametrize("times", [[100., 99.], [100., float("nan")],
                                  [100., float("inf")], [-1., 0.]])
def test_invalid_real_event_times_are_rejected(times):
    with pytest.raises(ValueError):
        EHRDataset(records([11, 22], times), time_mode="time_conditioned")[0]


def test_repeated_timestamps_are_valid_and_first_sos_gap_is_zero():
    item = EHRDataset(records([11, 22, 33], [100., 100., 100.]),
                      max_length=4, time_mode="time_conditioned")[0]
    assert torch.equal(item["input_time"][:3], item["target_time"][:3])
    model = make_model().eval()
    with torch.no_grad():
        output = model(
            item["input_ids"].unsqueeze(0),
            input_time=item["input_time"].unsqueeze(0),
            target_time=item["target_time"].unsqueeze(0),
            y=item["labels"].unsqueeze(0),
            retention_mask=item["attention_mask"].unsqueeze(0),
        )
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["logits"]).all()


def test_historical_spacing_changes_prediction_with_the_final_query_gap_fixed():
    model = make_model().eval()
    first = example()
    second = {key: tensor.clone() for key, tensor in first.items()}
    second["input_time"][0, 2] = 108.
    second["target_time"][0, 1] = 108.
    # At the last output both conditions still have time 110 -> target time 112.
    assert first["input_time"][0, -1] == second["input_time"][0, -1]
    assert first["target_time"][0, -1] == second["target_time"][0, -1]
    with torch.no_grad():
        before = model(**first, output_hidden_states=True)
        after = model(**second, output_hidden_states=True)
    assert not torch.allclose(before["hidden_states"][-1][:, -1],
                              after["hidden_states"][-1][:, -1], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(before["logits"][:, -1], after["logits"][:, -1],
                              atol=1e-6, rtol=1e-5)


def test_query_time_changes_only_its_output_not_history_representations():
    model = make_model().eval()
    first = example()
    second = {key: tensor.clone() for key, tensor in first.items()}
    second["target_time"][0, -1] = 140.
    with torch.no_grad():
        before = model(**first, output_hidden_states=True)
        after = model(**second, output_hidden_states=True)
    for previous, changed in zip(before["hidden_states"], after["hidden_states"]):
        torch.testing.assert_close(previous, changed, atol=0, rtol=0)
    torch.testing.assert_close(before["logits"][:, :-1], after["logits"][:, :-1],
                               atol=0, rtol=0)
    assert not torch.allclose(before["logits"][:, -1], after["logits"][:, -1],
                              atol=1e-6, rtol=1e-5)


def test_future_token_and_history_time_do_not_change_earlier_outputs():
    model = make_model().eval()
    first = example()
    first["target_time"][0, -1] = 140.
    second = {key: tensor.clone() for key, tensor in first.items()}
    second["X"][0, -1] = 44
    second["input_time"][0, -1] = 130.
    with torch.no_grad():
        before = model(**first)
        after = model(**second)
    torch.testing.assert_close(before["logits"][:, :-1], after["logits"][:, :-1],
                               rtol=1e-5, atol=1e-6)


def test_masked_padding_times_cannot_change_valid_logits_or_loss():
    model = make_model().eval()
    short = example()
    padded = {
        "X": torch.cat([short["X"], torch.zeros(1, 3, dtype=torch.long)], dim=1),
        "input_time": torch.cat([short["input_time"],
                                 torch.tensor([[float("nan"), float("inf"), -999.]])], dim=1),
        "target_time": torch.cat([short["target_time"],
                                  torch.tensor([[float("inf"), float("nan"), -1000.]])], dim=1),
        "y": torch.cat([short["y"], torch.zeros(1, 3, dtype=torch.long)], dim=1),
        "retention_mask": torch.cat([short["retention_mask"],
                                     torch.zeros(1, 3, dtype=torch.bool)], dim=1),
    }
    with torch.no_grad():
        before = model(**short)
        after = model(**padded)
    assert torch.isfinite(after["logits"]).all()
    torch.testing.assert_close(before["logits"], after["logits"][:, :4],
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(before["loss"], after["loss"], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("missing", ["input_time", "target_time"])
def test_temporal_model_requires_both_time_inputs(missing):
    model = make_model()
    batch = example()
    batch.pop(missing)
    with pytest.raises(ValueError, match="time"):
        model(**batch)


def test_query_before_latest_history_is_rejected():
    model = make_model()
    batch = example()
    batch["target_time"][0, -1] = 109.
    with pytest.raises(ValueError):
        model(**batch)


def test_global_time_origin_shift_and_fractional_large_ages_are_supported():
    model = make_model().eval()
    first = example()
    first["input_time"] += 0.25
    first["target_time"] += 0.25
    shifted = {key: tensor.clone() for key, tensor in first.items()}
    shifted["input_time"] += 1_000_000.
    shifted["target_time"] += 1_000_000.
    with torch.no_grad():
        before = model(**first)
        after = model(**shifted)
    assert torch.isfinite(after["loss"])
    torch.testing.assert_close(before["logits"], after["logits"], rtol=1e-5, atol=1e-6)


def test_target_condition_and_backbone_learn_while_embeddings_stay_fixed():
    model = make_model().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    fixed = model.token_embedding.emb.weight.detach().clone()
    projection = model.target_time_projection.weight.detach().clone()
    result = model(**example())
    result["loss"].backward()
    assert model.target_time_projection.weight.grad.abs().sum() > 0
    assert model.SRA_blocks[0].sra.qkv.weight.grad.abs().sum() > 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    optimizer.step()
    assert not torch.equal(projection, model.target_time_projection.weight)
    torch.testing.assert_close(fixed, model.token_embedding.emb.weight, rtol=0, atol=0)
    assert not model.token_embedding.emb.weight.requires_grad


def test_activation_checkpointing_preserves_loss_and_parameter_gradients():
    reference = make_model(num_layers=2).train()
    checkpointed = make_model(num_layers=2, use_grad_ckp=True).train()
    checkpointed.load_state_dict(reference.state_dict(), strict=True)
    first = reference(**example())
    second = checkpointed(**example())
    torch.testing.assert_close(first["loss"], second["loss"], rtol=1e-6, atol=1e-7)
    first["loss"].backward()
    second["loss"].backward()
    for (name, parameter), (other_name, other) in zip(
            reference.named_parameters(), checkpointed.named_parameters()):
        assert name == other_name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode", ["time_conditioned", "event_order"])
def test_checkpoint_preserves_temporal_mode_and_exact_predictions(tmp_path, mode):
    model = make_model(time_mode=mode).eval()
    config = vars(make_config(time_mode=mode))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint = tmp_path / (mode + ".pt")
    save_checkpoint(checkpoint, model, optimizer, config,
                    {"train": [1], "valid": [2], "test": [3]},
                    epoch=1, global_step=2, best_val_loss=1.0)
    loaded, payload = load_checkpoint(checkpoint, device="cpu")
    loaded.eval()
    assert payload["config"]["time_mode"] == mode
    if mode == "time_conditioned":
        assert loaded.time_mode == "time_conditioned"
        assert payload["config"]["time_unit"] == "months"
        assert payload["config"]["time_scale"] == 1.0
        inputs = example()
    else:
        inputs = {"X": example()["X"], "y": example()["y"]}
    with torch.no_grad():
        before = model(**inputs)
        after = loaded(**inputs)
    torch.testing.assert_close(before["logits"], after["logits"], rtol=0, atol=0)


def test_checkpoint_records_the_exact_temporal_training_contract(tmp_path):
    model = make_model(time_scale=2.5).eval()
    config = vars(make_config(time_scale=2.5))
    original_config = dict(config)
    checkpoint = tmp_path / "temporal_metadata.pt"
    save_checkpoint(checkpoint, model, None, config,
                    {"train": [1], "valid": [2], "test": [3]},
                    epoch=3, global_step=7, best_val_loss=1.25)
    loaded, payload = load_checkpoint(checkpoint, device="cpu")
    expected = {
        "time_mode": "time_conditioned",
        "time_unit": "months",
        "time_scale": 2.5,
        "history_time_encoding": "temporal_rope_headwise_v1",
        "target_time_encoding": "log1p_delta_add_v1",
        "sos_time_policy": "first_event",
        "pretraining_objective": "time_conditioned_next_event_v1",
    }
    assert {key: payload["config"][key] for key in expected} == expected
    assert config == original_config  # Serialization must not mutate the caller.
    assert loaded.time_mode == "time_conditioned"
    assert loaded.time_scale == 2.5
    assert payload["epoch"] == 3
    assert payload["global_step"] == 7


@pytest.mark.parametrize("override", [
    ["--time_mode", "event_order"],
    ["--time_scale", "2"],
])
def test_resume_rejects_time_contract_changes_before_reading_data(tmp_path, monkeypatch, override):
    import run_pretrain

    model = make_model()
    checkpoint = tmp_path / "source.pt"
    save_checkpoint(checkpoint, model, None, vars(make_config()),
                    {"train": [1], "valid": [2], "test": [3]},
                    epoch=1, global_step=0, best_val_loss=1.0)
    original_bytes = checkpoint.read_bytes()
    outputs = tmp_path / "runs"

    def unexpected_data_read(*args, **kwargs):
        pytest.fail("Resume contract validation must precede loading patient data")

    monkeypatch.setattr(run_pretrain, "data_provider", unexpected_data_read)
    with pytest.raises(ValueError, match="Cannot change time_"):
        run_pretrain.main([
            "--resume", str(checkpoint), "--device", "cpu",
            "--checkpoints", str(outputs), "--run_name", "must_not_exist",
        ] + override)
    assert checkpoint.read_bytes() == original_bytes
    assert not outputs.exists()


def test_batched_patients_keep_independent_time_origins_intervals_and_padding():
    model = make_model().eval()
    first = example()
    second = {
        "X": torch.tensor([[-1, 44, 22]]),
        "input_time": torch.tensor([[10_000., 10_000., 10_000.5]]),
        "target_time": torch.tensor([[10_000., 10_000.5, 10_020.]]),
        "y": torch.tensor([[44, 22, 11]]),
        "retention_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    pad_values = {
        "X": 0, "input_time": float("nan"), "target_time": float("inf"),
        "y": 0, "retention_mask": False,
    }
    batched = {
        key: torch.cat([
            first[key],
            torch.cat([value, value.new_full((1, 1), pad_values[key])], dim=1),
        ], dim=0)
        for key, value in second.items()
    }
    with torch.no_grad():
        first_output = model(**first)
        second_output = model(**second)
        batch_output = model(**batched)

    torch.testing.assert_close(batch_output["logits"][0], first_output["logits"][0],
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(batch_output["logits"][1, :3], second_output["logits"][0],
                               rtol=1e-5, atol=1e-6)
    # The batch mean must weight each patient's contribution by real target count.
    expected_loss = (4 * first_output["loss"] + 3 * second_output["loss"]) / 7
    torch.testing.assert_close(batch_output["loss"], expected_loss,
                               rtol=1e-5, atol=1e-6)
