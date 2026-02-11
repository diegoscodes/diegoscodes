"""Utility helpers for generating SHAP dependence plots.

This module provides a small wrapper around :func:`shap.dependence_plot`
that guards against a common pitfall: passing arrays with different numbers
of rows for the SHAP values and the feature matrix.  When the number of
samples does not match the scatter plot inside SHAP raises a ``ValueError``
complaining that ``x`` and ``y`` must be the same size.  The helper below
aligns the inputs automatically so the original analysis script can stay
focused on the modelling logic.

Example
-------

>>> aligned_dependence_plot(
...     "Year", sv_pos, X_test_trans,
...     feature_names=feature_names,
...     interaction_index=None,
... )

The wrapper trims the larger input so that SHAP receives matching arrays and
therefore produces the dependence plot without raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import shap


@dataclass
class _AlignedInputs:
    """Container holding SHAP values and features with matching row counts."""

    shap_values: np.ndarray
    features: pd.DataFrame
    feature_names: Sequence[str]


def _extract_shap_array(shap_values: object) -> np.ndarray:
    """Return the SHAP values as a 2D NumPy array.

    Parameters
    ----------
    shap_values:
        Object returned by SHAP explainers.  The function understands raw
        ``numpy`` arrays as well as ``shap.Explanation`` objects (which expose
        the ``values`` attribute).
    """

    if hasattr(shap_values, "values"):
        values = np.asarray(getattr(shap_values, "values"))
    else:
        values = np.asarray(shap_values)

    if values.ndim == 1:
        # Dependence plots require a column per feature.  When the caller
        # already selected a single feature we keep the semantics by treating
        # the values as a single-column matrix.
        values = values.reshape(-1, 1)

    if values.ndim != 2:
        raise ValueError(
            "SHAP values must be convertible to a 2D array of shape "
            "(n_samples, n_features)."
        )

    return values


def _coerce_features(
    features: object, feature_names: Optional[Sequence[str]]
) -> Tuple[pd.DataFrame, Sequence[str]]:
    """Return the feature matrix as a :class:`pandas.DataFrame`.

    The function keeps the existing column order when a dataframe is supplied
    and generates surrogate names when necessary.
    """

    if isinstance(features, pd.DataFrame):
        df = features.copy()
        resolved_names: Sequence[str] = list(df.columns)
    else:
        array = np.asarray(features)
        if array.ndim != 2:
            raise ValueError(
                "Feature matrix must be 2-dimensional; received shape "
                f"{array.shape}."
            )

        if feature_names is None:
            resolved_names = [f"feature_{i}" for i in range(array.shape[1])]
        else:
            if len(feature_names) != array.shape[1]:
                raise ValueError(
                    "The number of provided feature names does not match the "
                    "width of the feature matrix."
                )
            resolved_names = list(feature_names)

        df = pd.DataFrame(array, columns=resolved_names)

    return df, resolved_names


def _align_inputs(
    shap_values: np.ndarray, features: pd.DataFrame, feature_names: Sequence[str]
) -> _AlignedInputs:
    """Ensure the SHAP values and feature matrix contain the same samples."""

    n_shap, n_features = shap_values.shape
    if n_features != features.shape[1]:
        raise ValueError(
            "The number of feature columns in the SHAP values "
            f"({n_features}) does not match the feature matrix "
            f"({features.shape[1]})."
        )

    n_samples = len(features)
    if n_samples == n_shap:
        return _AlignedInputs(shap_values, features, feature_names)

    min_samples = min(n_shap, n_samples)
    trimmed_shap = shap_values[:min_samples]
    trimmed_features = features.iloc[:min_samples].reset_index(drop=True)

    return _AlignedInputs(trimmed_shap, trimmed_features, feature_names)


def aligned_dependence_plot(
    feature: str,
    shap_values: object,
    features: object,
    *,
    feature_names: Optional[Sequence[str]] = None,
    interaction_index: Optional[Union[int, str, Sequence[int]]] = None,
    show: bool = True,
) -> None:
    """Safely dispatch to :func:`shap.dependence_plot`.

    The helper mirrors SHAP's interface but avoids the ``ValueError`` raised by
    matplotlib when the two inputs describe different numbers of samples.  When
    that happens the larger input is truncated so that both contain the same
    number of rows before calling SHAP.
    """

    shap_array = _extract_shap_array(shap_values)
    feature_frame, resolved_names = _coerce_features(features, feature_names)
    aligned = _align_inputs(shap_array, feature_frame, resolved_names)

    shap.dependence_plot(
        feature,
        aligned.shap_values,
        aligned.features,
        feature_names=aligned.feature_names,
        interaction_index=interaction_index,
        show=show,
    )


__all__ = ["aligned_dependence_plot"]
