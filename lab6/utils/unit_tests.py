# tests_monitoring_light.py
# -*- coding: utf-8 -*-
from typing import Callable, Dict
import pandas as pd
import numpy as np


def test_parse_timestamps_adds_ts(
    df: pd.DataFrame,
    parse_timestamps: Callable[[pd.DataFrame], None],
) -> None:
    """
    Assert that parse_timestamps adds a datetime64 'ts' column.
    Usage:
        test_parse_timestamps_adds_ts(df, parse_timestamps)
    """
    work = df.copy()
    parse_timestamps(work)
    assert "ts" in work.columns, "Column 'ts' not found after parse_timestamps()"
    assert np.issubdtype(work["ts"].dtype, np.datetime64), "'ts' must be datetime dtype"


def test_normalize_errors_column(
    normalize_errors: Callable[[pd.DataFrame], None],
) -> None:
    """
    Assert that normalize_errors creates a '_error_norm' string column
    and maps empty string to 'none'.
    Usage:
        test_normalize_errors_column(normalize_errors)
    """
    tmp = pd.DataFrame({"error_type": ["", "knowledge_outdated", "json_format_error"]})
    normalize_errors(tmp)
    assert "_error_norm" in tmp.columns, "Column '_error_norm' not found after normalize_errors()"
    assert tmp["_error_norm"].map(type).eq(str).all(), "_error_norm must be strings"
    assert tmp.loc[0, "_error_norm"] == "none"
    assert tmp.loc[1, "_error_norm"] == "knowledge_outdated"
    assert tmp.loc[2, "_error_norm"] == "json_format_error"


def test_compute_metrics_returns_keys(
    df: pd.DataFrame,
    compute_metrics: Callable[[pd.DataFrame], Dict[str, float]],
) -> None:
    """
    Assert that compute_metrics returns required keys with float values.
    Usage:
        test_compute_metrics_returns_keys(df, compute_metrics)
    """
    m = compute_metrics(df)
    required = ["avg_latency_ms", "p95_latency_ms", "avg_tokens", "error_rate_pct", "avg_satisfaction"]
    for k in required:
        assert k in m, f"Missing metric key: {k}"
        assert isinstance(m[k], float), f"Metric {k} must be float"


def test_compute_metrics_reasonable_ranges(
    df: pd.DataFrame,
    compute_metrics: Callable[[pd.DataFrame], Dict[str, float]],
) -> None:
    """
    Assert that compute_metrics values are in reasonable ranges.
    Usage:
        test_compute_metrics_reasonable_ranges(df, compute_metrics)
    """
    m = compute_metrics(df)
    assert m["avg_latency_ms"] > 0
    assert m["p95_latency_ms"] > 0
    assert m["avg_tokens"] > 0
    assert 0.0 <= m["error_rate_pct"] <= 100.0
    assert 1.0 <= m["avg_satisfaction"] <= 5.0


def test_check_alerts_high_is_worse(
    check_alerts: Callable[[Dict[str, float], Dict[str, float]], list[str]]
) -> None:
    """
    Assert that higher values trigger alerts (except satisfaction which is above the min).
    Usage:
        test_check_alerts_high_is_worse(check_alerts)
    """
    metrics = {
        "avg_latency_ms": 3500.0,
        "p95_latency_ms": 7000.0,
        "error_rate_pct": 9.0,
        "avg_tokens": 1500.0,
        "avg_satisfaction": 4.1,
    }
    thresholds = {
        "avg_latency_ms": 3000.0,
        "p95_latency_ms": 5000.0,
        "error_rate_pct": 5.0,
        "avg_tokens": 1200.0,
        "avg_satisfaction_min": 3.0,
    }
    alerts = check_alerts(metrics, thresholds)
    assert any("avg_latency_ms" in a for a in alerts)
    assert any("p95_latency_ms" in a for a in alerts)
    assert any("error_rate_pct" in a for a in alerts)
    assert any("avg_tokens" in a for a in alerts)
    assert not any("avg_satisfaction" in a for a in alerts), "satisfaction above min → should NOT alert"


def test_check_alerts_low_is_worse_satisfaction(
    check_alerts: Callable[[Dict[str, float], Dict[str, float]], list[str]]
) -> None:
    """
    Assert that low satisfaction below the threshold triggers exactly one alert.
    Usage:
        test_check_alerts_low_is_worse_satisfaction(check_alerts)
    """
    metrics = {
        "avg_latency_ms": 1000.0,
        "p95_latency_ms": 2000.0,
        "error_rate_pct": 1.0,
        "avg_tokens": 500.0,
        "avg_satisfaction": 2.5,
    }
    thresholds = {
        "avg_latency_ms": 3000.0,
        "p95_latency_ms": 5000.0,
        "error_rate_pct": 5.0,
        "avg_tokens": 1200.0,
        "avg_satisfaction_min": 3.0,
    }
    alerts = check_alerts(metrics, thresholds)
    assert len(alerts) == 1, "Only satisfaction should trigger here"
    assert "avg_satisfaction" in alerts[0]
