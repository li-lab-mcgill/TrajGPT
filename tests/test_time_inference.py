"""Synthetic checks for time-conditioned autoregressive inference and evaluation."""

from types import MethodType, SimpleNamespace

import pytest
import torch

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


def recording_model(**overrides):
    model = make_model(**overrides)
    model.calls = []

    def forward(self, X, input_time=None, target_time=None, y=None, **kwargs):
        assert y is None, "True future diagnoses must never enter inference."
        assert not self.training
        assert torch.is_inference_mode_enabled()
        assert kwargs.get("past_key_values") is None
        assert kwargs.get("forward_impl", "parallel") == "parallel"
        assert kwargs["retention_mask"].all()
        assert X.ne(0).all()
        self.calls.append({
            "codes": X[0].detach().cpu().tolist(),
            "input_time": input_time[0].detach().cpu().tolist(),
            "target_time": target_time[0].detach().cpu().tolist(),
        })
        logits = torch.zeros((*X.shape, len(CODES) + 1), device=X.device)
        logits[..., -1] = 100.0  # PAD is highest, but must never be emitted.
        index = (CODES.index(int(X[0, -1])) + 1) % len(CODES)
        logits[0, -1, index] = 10.0
        return {"logits": logits, "loss": None}

    model.forward = MethodType(forward, model)
    return model


def test_feedback_keeps_event_times_aligned_and_never_shifts_latest_event():
    model = recording_model(seq_len=4).train()
    history = [11, 22, 33]
    times = [101., 103., 110.]
    future = [120., 120., 131.]
    result = model.inference(history, times, future, top_k=99)

    assert result["phecodes"] == [44, 11, 22]
    assert result["times"] == future
    assert model.calls == [
        {"codes": [-1, 11, 22, 33],
         "input_time": [101., 101., 103., 110.],
         "target_time": [101., 103., 110., 120.]},
        {"codes": [-1, 22, 33, 44],
         "input_time": [103., 103., 110., 120.],
         "target_time": [103., 110., 120., 120.]},
        {"codes": [-1, 33, 44, 11],
         "input_time": [110., 110., 120., 120.],
         "target_time": [110., 120., 120., 131.]},
    ]
    assert history == [11, 22, 33]
    assert times == [101., 103., 110.]
    assert future == [120., 120., 131.]
    assert model.training
    for code, candidates, probabilities in zip(
            result["phecodes"], result["top_k_phecodes"],
            result["top_k_probabilities"]):
        assert candidates[0] == code
        assert candidates[1:] == [candidate for candidate in CODES if candidate != code]
        assert set(candidates) == set(CODES)
        assert len(probabilities) == len(CODES)
        assert sum(probabilities) == pytest.approx(1.0)
        assert probabilities == sorted(probabilities, reverse=True)


def test_long_prefix_keeps_recent_events_and_single_dimension_tensors_work():
    model = recording_model(seq_len=4)
    output = model.inference(
        torch.tensor([11, 22, 33, 44]),
        torch.tensor([100., 101., 105., 110.]),
        torch.tensor([112.]), context_length=3,
    )
    assert output["phecodes"] == [11]
    assert model.calls[0] == {
        "codes": [-1, 33, 44],
        "input_time": [105., 105., 110.],
        "target_time": [105., 110., 112.],
    }


def test_empty_future_returns_empty_sequence_without_forward():
    model = recording_model()
    output = model.inference([11], [100.], [])
    assert output == {"phecodes": [], "times": [], "top_k_phecodes": [],
                      "top_k_probabilities": []}
    assert not model.calls


@pytest.mark.parametrize("changes", [
    {"history_codes": []},
    {"history_codes": [11, 0]},
    {"history_codes": [11, -1]},
    {"history_codes": [11, 99]},
    {"history_codes": [11, 22.5]},
    {"history_codes": [[11, 22]]},
    {"history_times": [100.]},
    {"history_times": [100., 99.]},
    {"history_times": [-1., 100.]},
    {"history_times": [100., float("nan")]},
    {"future_times": [100.]},
    {"future_times": [120., 119.]},
    {"future_times": [float("inf")]},
    {"future_times": [[120.]]},
    {"context_length": 1},
    {"context_length": 5},
    {"top_k": 0},
])
def test_invalid_input_is_rejected_before_forward(changes):
    model = recording_model(seq_len=4)
    arguments = dict(history_codes=[11, 22], history_times=[100., 110.],
                     future_times=[120.])
    arguments.update(changes)
    with pytest.raises((ValueError, TypeError)):
        model.inference(**arguments)
    assert not model.calls


def test_event_order_model_cannot_silently_ignore_time_queries():
    with pytest.raises(ValueError):
        make_model(time_mode="event_order").inference([11], [100.], [110.])


def test_nonfinite_prediction_fails_and_restores_training_mode():
    model = make_model().train()

    def broken_forward(self, X, **kwargs):
        assert not self.training
        return {"logits": torch.full((*X.shape, 5), float("nan"))}

    model.forward = MethodType(broken_forward, model)
    with pytest.raises((ValueError, RuntimeError)):
        model.inference([11], [100.], [110.])
    assert model.training


def test_real_model_inference_is_deterministic_and_does_not_update_weights():
    model = make_model(dropout=0.4).train()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    arguments = dict(history_codes=[11, 22, 33],
                     history_times=[100., 102., 110.],
                     future_times=[110., 117., 139.], top_k=3)
    first = model.inference(**arguments)
    second = model.inference(**arguments)
    assert first == second
    assert len(first["phecodes"]) == 3
    assert first["times"] == arguments["future_times"]
    assert all(code in CODES for code in first["phecodes"])
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)


def test_reloaded_temporal_checkpoint_produces_identical_rollout(tmp_path):
    model = make_model(seq_len=4)
    arguments = dict(history_codes=[11, 22, 33],
                     history_times=[100., 102., 110.],
                     future_times=[110., 117., 139.])
    expected = model.inference(**arguments)
    path = tmp_path / "temporal.pt"
    save_checkpoint(
        path, model, None, vars(make_config(seq_len=4)),
        {"train": ["train"], "valid": ["valid"], "test": ["test"]},
        epoch=3, global_step=57, best_val_loss=5.2,
    )
    loaded, payload = load_checkpoint(path)
    assert loaded.inference(**arguments) == expected
    assert payload["config"]["time_mode"] == "time_conditioned"


def test_evaluation_compares_aligned_events_and_reports_top_k():
    from run_inference import evaluate_predictions

    predictions = {
        "phecodes": [11, 22, 44], "times": [110., 120., 120.],
        "top_k_phecodes": [[11, 22], [22, 33], [44, 11]],
        "top_k_probabilities": [[.8, .1], [.6, .2], [.7, .1]],
    }
    metrics = evaluate_predictions(predictions, [11, 33, 22])
    assert metrics["n_events"] == 3
    assert metrics["correct"] == 1
    assert metrics["top1_accuracy"] == pytest.approx(1 / 3)
    assert metrics["top_k"] == 2
    assert metrics["top_k_correct"] == 2
    assert metrics["top_k_accuracy"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        evaluate_predictions(predictions, [11, 33])


def test_evaluator_uses_first_fifty_and_true_future_codes_cannot_affect_rollout():
    from run_inference import evaluate_patient

    prefix = [CODES[index % 4] for index in range(50)]
    times = [100. + index * index for index in range(52)]
    first_model = recording_model()
    first = evaluate_patient(first_model, prefix + [11, 11], times)
    second_model = recording_model()
    second = evaluate_patient(second_model, prefix + [33, 44], times)

    assert first["history_phecodes"] == prefix
    assert first["history_times"] == times[:50]
    assert first["future_times"] == times[50:]
    assert first["true_phecodes"] == [11, 11]
    assert first["predictions"]["phecodes"] == [33, 44]
    assert first["predictions"] == second["predictions"]
    assert first_model.calls == second_model.calls
    assert first["metrics"]["correct"] == 0
    assert second["metrics"]["correct"] == 2
    assert first_model.calls[0]["codes"] == [-1] + prefix
    assert first_model.calls[1]["codes"] == [-1] + prefix + [33]


def test_evaluation_horizon_slices_future_times_and_labels_together():
    from run_inference import evaluate_patient

    prefix = [CODES[index % 4] for index in range(50)]
    times = [100. + index for index in range(53)]
    model = recording_model()
    result = evaluate_patient(model, prefix + [33, 44, 11], times,
                              max_new_tokens=1)
    assert result["future_times"] == [150.]
    assert result["true_phecodes"] == [33]
    assert result["metrics"]["n_events"] == 1
    assert len(model.calls) == 1
