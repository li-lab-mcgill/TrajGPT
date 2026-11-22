"""PopHR batches with explicit teacher forcing and right padding.

The existing 10,000-row subset is intentionally retained. A sequence window
contains at most ``max_length`` target events. Its training input is SOS followed
by all but the final target, so the latest event remains the final label.
"""

import math
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = 0
SOS_TOKEN = -1
DATA_PATH = Path(__file__).resolve().parent / "processed_pophr_data.csv"
DATA_ROW_LIMIT = 10000


class EHRDataset(Dataset):
    def __init__(self, data, max_length=256, time_mode="event_order"):
        if not isinstance(max_length, int) or max_length < 2:
            raise ValueError("max_length must be an integer of at least 2")
        self.max_length = max_length
        if time_mode not in ("event_order", "time_conditioned"):
            raise ValueError("time_mode must be event_order or time_conditioned")
        self.time_mode = time_mode
        self.data = self._preprocess_data(data)

    def _preprocess_data(self, data):
        required = ["id", "date", "PheCode", "age_at_diag"]
        missing = set(required) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required PopHR columns: {sorted(missing)}")
        if data.empty:
            raise ValueError("EHRDataset requires at least one clinical event")
        if data[required].isna().any().any():
            raise ValueError("Required PopHR columns contain missing values")

        # The explicit tie-breaker preserves original order for same-date events.
        ordered = data.copy()
        ordered["_event_order"] = range(len(ordered))
        ordered["_event_date"] = pd.to_datetime(ordered["date"], errors="raise")
        ordered = ordered.sort_values(
            ["id", "_event_date", "_event_order"], kind="stable"
        )
        # Normalize and validate each record once, not on every epoch access.
        codes = ordered["PheCode"].map(float)
        if any(not math.isfinite(code) or code <= 0 or not code.is_integer() for code in codes):
            raise ValueError("Clinical PheCodes must be positive integer codes")
        ordered["PheCode"] = codes.map(int)
        ordered["age_at_diag"] = ordered["age_at_diag"].map(float)
        if not all(map(math.isfinite, ordered["age_at_diag"])):
            raise ValueError("Clinical event times must be finite")
        if self.time_mode == "time_conditioned":
            if (ordered["age_at_diag"] < 0).any():
                raise ValueError("Clinical ages in months must be nonnegative")
            if (ordered.groupby("id")["age_at_diag"].diff() < 0).any():
                raise ValueError("Clinical event times must be nondecreasing; equal times are allowed")
        aggregations = {"PheCode": list, "age_at_diag": list}
        if "month_of_birth" in ordered.columns:
            aggregations["month_of_birth"] = "first"
        return ordered.groupby("id", sort=True).agg(aggregations).reset_index()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient = self.data.iloc[idx]
        # Slice both original sequences together before adding SOS or padding.
        codes = patient.PheCode[-self.max_length :]
        times = patient.age_at_diag[-self.max_length :]

        valid_length = len(codes)
        padding_length = self.max_length - valid_length
        input_ids = [SOS_TOKEN] + codes[:-1]
        # In time-conditioned mode SOS shares the first retained event's time.
        # Thus the first query has zero gap, without inventing a birth-to-event gap.
        sos_time = times[0] if self.time_mode == "time_conditioned" else 0.0
        input_time = [sos_time] + times[:-1]
        # Pad with the final input time; padding never represents a real event.
        padding_time = input_time[-1]
        result = {
            "input_ids": torch.tensor(
                input_ids + [PAD_TOKEN] * padding_length, dtype=torch.long
            ),
            "input_time": torch.tensor(
                input_time + [padding_time] * padding_length, dtype=torch.float32
            ),
            "labels": torch.tensor(
                codes + [PAD_TOKEN] * padding_length, dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [True] * valid_length + [False] * padding_length, dtype=torch.bool
            ),
        }
        if self.time_mode == "time_conditioned":
            result["target_time"] = torch.tensor(
                times + [times[-1]] * padding_length, dtype=torch.float32
            )
        return result


def _validate_split_ids(split_ids, patient_ids):
    names = ("train", "valid", "test")
    if not isinstance(split_ids, dict) or set(split_ids) != set(names):
        raise ValueError("split_ids must contain exactly train, valid, and test")
    result = {}
    seen = set()
    for name in names:
        if not isinstance(split_ids[name], (list, tuple)) or not split_ids[name]:
            raise ValueError(f"split_ids[{name!r}] must be a nonempty list")
        ids = [value.item() if hasattr(value, "item") else value for value in split_ids[name]]
        unique = set(ids)
        if len(unique) != len(ids):
            raise ValueError(f"Duplicate patient IDs within the {name} split")
        if seen & unique:
            raise ValueError("Patient splits must be disjoint")
        seen.update(unique)
        result[name] = ids
    if seen != set(patient_ids):
        raise ValueError("Saved splits do not exactly cover the current 10,000-row dataset")
    return result


def data_provider(
    batch_size,
    max_length=256,
    train_ratio=0.8,
    valid_ratio=0.1,
    split_ids=None,
    return_split_ids=False,
    time_mode="event_order",
):
    total_data = pd.read_csv(DATA_PATH, nrows=DATA_ROW_LIMIT)
    if total_data["id"].isna().any():
        raise ValueError("PopHR patient IDs must not be missing")
    patient_ids = total_data["id"].unique().tolist()

    if split_ids is None:
        if not (0 < train_ratio < 1 and 0 < valid_ratio < 1 - train_ratio):
            raise ValueError("Train, validation, and test ratios must all be positive")
        train_ids, temp_ids = train_test_split(
            patient_ids, test_size=1 - train_ratio, random_state=42
        )
        valid_ids, test_ids = train_test_split(
            temp_ids,
            test_size=1 - valid_ratio / (1 - train_ratio),
            random_state=42,
        )
        split_ids = {"train": train_ids, "valid": valid_ids, "test": test_ids}
    split_ids = _validate_split_ids(split_ids, patient_ids)

    result = []
    for name in ("train", "valid", "test"):
        split_data = total_data[total_data["id"].isin(split_ids[name])]
        dataset = EHRDataset(split_data, max_length=max_length, time_mode=time_mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=name == "train")
        result.extend((dataset, dataloader))
    if return_split_ids:
        result.append(split_ids)
    return tuple(result)
