import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Reproducibility seed for all numpy randomness
np.random.seed(42)

DATA_DIR = "data"
AEGIS_PATH = os.path.join(DATA_DIR, "aegis_raw_logs.csv")
NEXUS_PATH = os.path.join(DATA_DIR, "nexus_raw_telemetry.csv")
TOTAL_ROWS = 150_000


def ensure_data_dir():
    """Create the output directory if it does not exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def random_timestamps(n_rows, days_back):
    """Generate ISO timestamps uniformly over the last N days."""
    now = datetime.utcnow()
    max_seconds = days_back * 24 * 60 * 60
    offsets = np.random.randint(0, max_seconds, size=n_rows)
    return [(now - timedelta(seconds=int(s))).isoformat() for s in offsets]


def random_ipv4(n_rows):
    """Generate random IPv4 addresses."""
    octets = np.random.randint(1, 255, size=(n_rows, 4))
    return [".".join(map(str, parts)) for parts in octets]


def add_noise(values, min_value=None, is_int=False):
    """Apply bounded Gaussian noise (+/-5%) to numeric values."""
    values = values.astype(float)
    noise = np.random.normal(loc=0.0, scale=0.05, size=values.size)
    noise = np.clip(noise, -0.05, 0.05)
    adjusted = np.where(np.isnan(values), np.nan, values * (1.0 + noise))
    if min_value is not None:
        adjusted = np.where(np.isnan(adjusted), np.nan, np.maximum(adjusted, min_value))
    if is_int:
        adjusted = np.rint(adjusted)
        if min_value is not None:
            adjusted = np.where(adjusted < min_value, min_value, adjusted)
        return adjusted.astype(int)
    return adjusted


def oversample_minority(df, label_col, minority_label, numeric_spec):
    """Duplicate minority rows with small noise until minority reaches 20%."""
    counts = df[label_col].value_counts()
    minority_count = int(counts.get(minority_label, 0))
    majority_count = int(df.shape[0] - minority_count)
    target_minority = int(np.ceil(0.25 * majority_count))
    needed = max(0, target_minority - minority_count)
    if needed == 0:
        return df

    minority_df = df[df[label_col] == minority_label]
    sampled = minority_df.sample(n=needed, replace=True, random_state=42).copy()

    for col, spec in numeric_spec.items():
        sampled[col] = add_noise(
            sampled[col].to_numpy(),
            min_value=spec.get("min_value"),
            is_int=spec.get("is_int", False),
        )

    return pd.concat([df, sampled], ignore_index=True)


def finalize_dataset(df):
    """Shuffle and cap dataset to the required total row count."""
    return df.sample(frac=1.0, random_state=42).head(TOTAL_ROWS).reset_index(drop=True)


def build_aegis_dataset():
    """Generate the Aegis-Vanguard web logs dataset."""
    n_bot = int(TOTAL_ROWS * 0.05)
    n_human = TOTAL_ROWS - n_bot
    labels = np.array(["Human"] * n_human + ["Bot"] * n_bot)

    timestamps = random_timestamps(TOTAL_ROWS, days_back=30)
    ip_address = random_ipv4(TOTAL_ROWS)

    session_duration = np.random.exponential(scale=300, size=TOTAL_ROWS)
    null_count = int(TOTAL_ROWS * 0.02)
    null_idx = np.random.choice(TOTAL_ROWS, size=null_count, replace=False)
    session_duration[null_idx] = np.nan

    click_velocity = np.empty(TOTAL_ROWS)
    pages_viewed = np.empty(TOTAL_ROWS)

    human_mask = labels == "Human"
    bot_mask = ~human_mask

    click_velocity[human_mask] = np.random.normal(2.5, 0.5, size=human_mask.sum())
    click_velocity[bot_mask] = np.random.normal(18.0, 3.0, size=bot_mask.sum())

    pages_viewed[human_mask] = np.random.normal(4, 1, size=human_mask.sum())
    pages_viewed[bot_mask] = np.random.normal(35, 5, size=bot_mask.sum())

    click_velocity = np.clip(click_velocity, 0.1, None)
    pages_viewed = np.clip(np.rint(pages_viewed), 1, None).astype(int)

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
        "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
        "Mozilla/5.0 (iPad; CPU OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/121.0.0.0",
    ]
    ua_probs = [0.98 / 8.0] * 8 + [0.02]
    user_agent = np.random.choice(user_agents + ["Missing"], size=TOTAL_ROWS, p=ua_probs)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "ip_address": ip_address,
            "session_duration_sec": session_duration,
            "click_velocity_bps": click_velocity,
            "pages_viewed": pages_viewed,
            "user_agent": user_agent,
            "class_label": labels,
        }
    )

    numeric_spec = {
        "session_duration_sec": {"min_value": 1.0, "is_int": False},
        "click_velocity_bps": {"min_value": 0.1, "is_int": False},
        "pages_viewed": {"min_value": 1, "is_int": True},
    }

    df = oversample_minority(df, "class_label", "Bot", numeric_spec)
    df = finalize_dataset(df)
    return df


def build_nexus_dataset():
    """Generate the Nexus-Grid smart meter telemetry dataset."""
    n_overload = int(TOTAL_ROWS * 0.02)
    n_normal = TOTAL_ROWS - n_overload
    labels = np.array(["Normal"] * n_normal + ["Overload"] * n_overload)

    timestamps = np.array(random_timestamps(TOTAL_ROWS, days_back=30), dtype=object)
    corrupt_count = int(TOTAL_ROWS * 0.01)
    corrupt_idx = np.random.choice(TOTAL_ROWS, size=corrupt_count, replace=False)
    timestamps[corrupt_idx] = "CORRUPTED_TIME"

    sectors = ["F-6", "F-7", "Bahria-Phase-7", "Bahria-Phase-8", "DHA-1", "G-11"]
    sector_id = np.random.choice(sectors, size=TOTAL_ROWS)

    kw_draw = np.empty(TOTAL_ROWS)
    temperature_c = np.empty(TOTAL_ROWS)
    voltage_drop = np.empty(TOTAL_ROWS)

    normal_mask = labels == "Normal"
    overload_mask = ~normal_mask

    kw_draw[normal_mask] = np.random.normal(45, 8, size=normal_mask.sum())
    kw_draw[overload_mask] = np.random.normal(95, 5, size=overload_mask.sum())

    temperature_c[normal_mask] = np.random.normal(38, 5, size=normal_mask.sum())
    temperature_c[overload_mask] = np.random.normal(72, 4, size=overload_mask.sum())

    voltage_drop[normal_mask] = np.random.normal(5, 1, size=normal_mask.sum())
    voltage_drop[overload_mask] = np.random.normal(28, 3, size=overload_mask.sum())

    kw_draw = np.clip(kw_draw, 0.1, None)
    temperature_c = np.clip(temperature_c, -10.0, None)
    voltage_drop = np.clip(voltage_drop, 0.1, None)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "sector_id": sector_id,
            "kw_draw": kw_draw,
            "temperature_c": temperature_c,
            "voltage_drop": voltage_drop,
            "grid_status": labels,
        }
    )

    numeric_spec = {
        "kw_draw": {"min_value": 0.1, "is_int": False},
        "temperature_c": {"min_value": -10.0, "is_int": False},
        "voltage_drop": {"min_value": 0.1, "is_int": False},
    }

    df = oversample_minority(df, "grid_status", "Overload", numeric_spec)
    df = finalize_dataset(df)
    return df


def main():
    """Orchestrate dataset generation and write CSV outputs."""
    ensure_data_dir()

    aegis_df = build_aegis_dataset()
    nexus_df = build_nexus_dataset()

    aegis_df.to_csv(AEGIS_PATH, index=False)
    nexus_df.to_csv(NEXUS_PATH, index=False)


if __name__ == "__main__":
    main()
