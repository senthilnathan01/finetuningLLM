# util.py
# --------------------------------------------------------------------
# Semantic clustering utilities for failure cases
# --------------------------------------------------------------------
# HOW STUDENTS USE THIS:
#   from util import semantic_cluster_failure_cases
#   df_sem, info = semantic_cluster_failure_cases(failure_df)
#
# WHAT IT DOES:
#   1) Builds texts from (intent | user_prompt || llm_response || error_type)
#   2) Computes embeddings with SentenceTransformer (fallback to TF-IDF)
#   3) Picks k via silhouette, runs KMeans
#   4) Attaches labels as `semantic_cluster`
#
# YOU CAN TUNE:
#   - model_name: embedding model id (default: "all-MiniLM-L6-v2")
#   - k_range: search range for number of clusters
#   - use_error_hints: whether to include error_type in the text
#   - label_col: where to store cluster ids
#
# RETURNS:
#   df_with_labels, info_dict (contains k, labels, embeddings shape, etc.)
# --------------------------------------------------------------------

from __future__ import annotations

import re
import sys
import importlib
import subprocess
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display


# ----------------------------
# Check and install missing packages
# ----------------------------
def install_if_missing(package_name: str):
    """
    Check if a package is installed; if not, install it via pip.
    """
    try:
        importlib.import_module(package_name)
    except ImportError:
        print(f"⚠️ {package_name} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


# ----------------------------
# Text building for embeddings
# ----------------------------
def build_texts_for_embedding(
    df: pd.DataFrame,
    use_error_hints: bool = True,
) -> List[str]:
    """
    Build a single text string per row to feed into an embedding model.

    Format:
      "<intent> | <user_prompt> || <llm_response> [|| <error_type>]"

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least: 'intent', 'user_prompt', 'llm_response'.
        If use_error_hints=True, will also try to append 'error_type'.
    use_error_hints : bool
        If True, append error_type to provide extra context for clustering.

    Returns
    -------
    List[str]
        One combined text per row, length == len(df).
    """
    cols = ["intent", "user_prompt", "llm_response"]
    have_error = use_error_hints and ("error_type" in df.columns)

    texts: List[str] = []
    for _, r in df.iterrows():
        parts = [str(r.get(c, "")) for c in cols]
        # NOTE: Keep the separators consistent so the model sees structure.
        s = " | ".join([parts[0]]) + " || " + " || ".join(parts[1:])
        if have_error:
            s = f"{s} || {str(r.get('error_type', ''))}"
        # Normalize whitespace to avoid noise.
        s = re.sub(r"\s+", " ", s).strip()
        texts.append(s)
    return texts


# -----------------------
# Embedding computation
# -----------------------
def compute_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    use_tfidf_fallback: bool = True,
) -> np.ndarray:
    """
    Compute vector embeddings for the given texts.

    Strategy
    --------
    1) Try SentenceTransformer (best semantic quality).
    2) If unavailable (e.g., offline CI), fallback to TF-IDF to keep the lab runnable.

    Returns
    -------
    np.ndarray of shape (n_samples, dim)
    """
    try:
        # TIP: This import may fail in constrained environments.
        # In that case we fallback to TF-IDF vectors below.
        from sentence_transformers import SentenceTransformer
        import os
        
        # Try to load from pre-downloaded local model first
        # Path matches your actual model structure: models/sentence-transformers/all-MiniLM-L6-v2/
        local_model_path = f"/app/models/sentence-transformers/{model_name}"
        
        if os.path.exists(local_model_path):
            print(f"✅ Loading pre-downloaded model from: {local_model_path}")
            model = SentenceTransformer(local_model_path)
        else:
            print(f"⚠️ Local model not found at {local_model_path}, downloading from HuggingFace...")
            model = SentenceTransformer(model_name)
            
        print("🔄 Computing embeddings...")
        emb = model.encode(texts, show_progress_bar=False)
        print(f"✅ Embeddings computed: shape {emb.shape}")
        return np.asarray(emb, dtype=np.float32)
    except Exception:
        if not use_tfidf_fallback:
            raise
        # Fallback: deterministic, light-weight vectors
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
        X = vec.fit_transform(texts)
        return X.toarray().astype(np.float32)


# -----------------------
# k selection & KMeans
# -----------------------
def pick_k_by_silhouette(
    emb: np.ndarray,
    k_range: Tuple[int, int] = (2, 6),
) -> int:
    """
    Choose k by maximizing silhouette score over the given range.
    For tiny datasets, returns a small safe k.

    Returns
    -------
    int
        Chosen number of clusters.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = emb.shape[0]
    if n < 10:
        # Small-data safety: avoid degenerate silhouette computation
        return max(2, min(3, n // 3 + 1))

    best_k, best_score = None, -1.0
    low, high = k_range
    for k in range(low, min(high, n - 1) + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(emb)
        # silhouette needs at least two non-empty clusters
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(emb, labels)
        if score > best_score:
            best_k, best_score = k, score

    return best_k if best_k is not None else 2


def run_kmeans(
    emb: np.ndarray,
    k: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Run KMeans on embeddings.

    Returns
    -------
    np.ndarray
        Cluster labels of shape (n_samples,)
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(emb)
    return labels


# -----------------------
# Attach labels to frame
# -----------------------
def attach_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    label_col: str = "semantic_cluster",
) -> pd.DataFrame:
    """
    Return a copy of df with cluster labels attached.

    Raises
    ------
    AssertionError
        If len(labels) != len(df)
    """
    out = df.copy()
    assert len(out) == len(labels), "labels length must match df length"
    out[label_col] = labels
    return out


# -----------------------
# One-shot pipeline
# -----------------------
def semantic_cluster_failure_cases(
    df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    k_range: Tuple[int, int] = (2, 6),
    use_error_hints: bool = True,
    label_col: str = "semantic_cluster",
    use_tfidf_fallback: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    End-to-end semantic clustering for failure cases.

    Pipeline
    --------
    1) Build texts from (intent | user_prompt || llm_response [|| error_type])
    2) Compute embeddings (SentenceTransformer → TF-IDF fallback)
    3) Pick k by silhouette
    4) Run KMeans and attach labels as `label_col`

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of failure cases (e.g., filtered by satisfaction/thumbs/error in Part 3 Step 1).
    model_name : str
        SentenceTransformer model name. Change if you want different embeddings.
    k_range : (int, int)
        Range to search for k (inclusive bounds).
    use_error_hints : bool
        Whether to append error_type to the text input.
    label_col : str
        Column name to store the cluster IDs.
    use_tfidf_fallback : bool
        If True, fallback to TF-IDF when ST is unavailable.

    Returns
    -------
    (df_with_labels, info) : (pd.DataFrame, Dict[str, Any])
        df_with_labels : original df with an extra column = `label_col`
        info : dict with keys:
            - 'k' (int): chosen number of clusters
            - 'labels' (np.ndarray): cluster id per row
            - 'emb_shape' (tuple): embeddings shape
            - 'label_col' (str): the column name used for labels
            - 'model' (str): embedding backend used ("sentence-transformers" or "tfidf")
    """
    # 1) Build texts
    texts = build_texts_for_embedding(df, use_error_hints=use_error_hints)

    # 2) Embeddings (with safe fallback)
    try:
        emb = compute_embeddings(texts, model_name=model_name, use_tfidf_fallback=use_tfidf_fallback)
        backend = "sentence-transformers"
    except Exception as e:
        print(f"⚠️ SentenceTransformer failed: {e}")
        print("🔄 Falling back to TF-IDF vectorization...")
        # Defensive fallback even if use_tfidf_fallback=False was passed accidentally
        emb = compute_embeddings(texts, model_name=model_name, use_tfidf_fallback=True)
        backend = "tfidf"

    # 3) Pick k and 4) cluster
    k = pick_k_by_silhouette(emb, k_range=k_range)
    labels = run_kmeans(emb, k=k, random_state=42)

    # Attach
    df_out = attach_clusters(df, labels, label_col=label_col)

    info: Dict[str, Any] = {
        "k": int(k),
        "labels": labels,
        "emb_shape": tuple(emb.shape),
        "label_col": label_col,
        "model": backend,
    }
    return df_out, info


def plot_token_distribution(df, column="tokens_generated"):
    plt.figure(figsize=(6,4))
    plt.hist(df[column].dropna(), bins=30)
    plt.xlabel("Tokens Generated")
    plt.ylabel("Count")
    plt.title("Token Usage Distribution")
    plt.show()

def plot_error_types(df, column="error_type"):
    counts = df[column].value_counts()
    plt.figure(figsize=(6,4))
    counts.plot(kind="bar")
    plt.xlabel("Error Type")
    plt.ylabel("Count")
    plt.title("Error Type Distribution")
    plt.xticks(rotation=0)
    plt.show()

def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a datetime column 'ts' exists parsed from 'timestamp'.
    Returns the same DataFrame (mutates in place).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'timestamp' column
        
    Returns
    -------
    pd.DataFrame
        The same DataFrame with added 'ts' column containing parsed timestamps
    """
    if "ts" not in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"])
    return df


def normalize_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create/refresh '_error_norm' mapping empty string to 'none'.
    Returns the same DataFrame (mutates in place).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing an 'error_type' column
        
    Returns
    -------
    pd.DataFrame
        The same DataFrame with added '_error_norm' column where empty errors are mapped to 'none'
    """
    df["_error_norm"] = df["error_type"].fillna("").replace("", "none")
    return df


def plot_latency_distribution(df, column="response_time_ms"):
    """
    Plot the distribution of response latency (in milliseconds).
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset containing latency values
    column : str, optional
        The column name for latency, by default "response_time_ms"
    """
    # 1. Extract non-null values
    values = df[column].dropna()
    
    # 2. Plot histogram
    plt.figure(figsize=(6,4))
    plt.hist(values, bins=30, edgecolor="black")
    
    # 3. Add labels and title
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.title("Latency Distribution")
    plt.show()


def create_intervention_selector():
    """
    Create interactive widgets for selecting intervention strategies for different failure cases.
    
    This function creates a set of dropdown menus where users can select the most appropriate
    optimization technique for each failure case scenario, along with a submit button to 
    evaluate their choices.
    
    Returns
    -------
    None
        Displays the interactive widgets directly in the notebook
    """
    
    cases = {
        "Case 1: The bot gives a very long-winded answer to a simple FAQ.": [
            "RAG update", "Guardrails / Schema enforcement", "RL preference tuning", "Prompt adjustment"
        ],
        "Case 2: The bot outputs invalid JSON that breaks downstream parsers.": [
            "RAG update", "Guardrails / Schema enforcement", "RL preference tuning", "Prompt adjustment"
        ],
        "Case 3: The bot apologizes three times in one short answer.": [
            "RAG update", "Guardrails / Schema enforcement", "RL preference tuning", "Prompt adjustment"
        ],
        "Case 4: The bot still cites an old return policy from last year.": [
            "RAG update", "Guardrails / Schema enforcement", "RL preference tuning", "Prompt adjustment"
        ]
    }

    default_choices = [
        "Prompt adjustment",                     # Case 1
        "Guardrails / Schema enforcement",       # Case 2
        "RL preference tuning",                  # Case 3
        "RAG update"                             # Case 4
    ]

    # Create dropdowns and store them
    dropdowns = []
    for i, (q, opts) in enumerate(cases.items()):
        print(q)
        dd = widgets.Dropdown(
            options=opts,
            value=default_choices[i],
            description="Your choice:",
            layout=widgets.Layout(width='70%')
        )
        display(dd)
        dropdowns.append(dd)

    # Add a submit button and output area
    button = widgets.Button(
        description="Submit All Choices",
        button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
        layout=widgets.Layout(width='200px', margin='20px 0')
    )
    output = widgets.Output()

    def on_submit_click(b):
        with output:
            output.clear_output()
            print("=" * 50)
            print("📊 Your Selected Interventions:")
            print("=" * 50)
            
            # Expected answers for comparison (optional)
            expected = {
                "Case 1": "Prompt adjustment",
                "Case 2": "Guardrails / Schema enforcement", 
                "Case 3": "RL preference tuning",
                "Case 4": "RAG update"
            }
            
            correct_count = 0
            for i, (case_name, _) in enumerate(cases.items()):
                case_num = case_name.split(":")[0]
                selected = dropdowns[i].value
                
                # Check if it matches expected (optional feedback)
                if selected == expected[case_num]:
                    status = "✅"
                    correct_count += 1
                else:
                    status = "🤔"
                
                print(f"{status} {case_num}: {selected}")
            
            print("\n" + "=" * 50)
            print(f"Score: {correct_count}/4 matches with common solutions")
            print("\nConsider: Are there cases where alternative approaches might also work?")

    # Display button and output
    display(button)
    display(output)

    # Attach the event handler
    button.on_click(on_submit_click)


def create_priority_decision_selector():
    """
    Create an interactive widget for selecting the highest priority intervention strategy.
    
    This function creates a dropdown menu where users can select which optimization
    technique they would prioritize first for deployment in production, along with
    a submit button to confirm their choice.
    
    Returns
    -------
    None
        Displays the interactive widgets directly in the notebook
    """
    
    decision_options = [
        "RAG update (fix knowledge freshness)",
        "Guardrails / Schema enforcement (fix JSON errors)",
        "RL preference tuning (reduce verbosity, adjust tone)",
        "Prompt/template adjustments (quick fixes for style/length)"
    ]

    decision = widgets.Dropdown(
        options=decision_options,
        description="Choose intervention:",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )

    # Use a button for more reliable triggering
    button = widgets.Button(description="Submit Choice")
    output = widgets.Output()

    def on_button_click(b):
        with output:
            output.clear_output()
            print(f"✅ You selected: {decision.value}")
            print("Now justify: Why would this be your first priority in production?")

    button.on_click(on_button_click)

    # Display everything together
    display(decision)
    display(button)
    display(output)
