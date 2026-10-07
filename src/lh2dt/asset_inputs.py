"""Private facility-configuration contracts with parameter provenance.

The public library does not ship TK-1101/TK-1102 values.  A facility owner
can keep a JSON (or optional PyYAML) file outside the repository and use these
helpers to validate that every model parameter has a unit, source, uncertainty
and applicability statement before constructing a model with the existing
catalog/factory functions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ParameterEvidence:
    value: Any
    unit: str
    source: str
    uncertainty: float | None = None
    applies_to: str = ""
    valid_range: tuple[float, float] | None = None
    notes: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, name: str) -> "ParameterEvidence":
        if not isinstance(value, Mapping):
            raise ValueError(f"evidence for {name} must be an object")
        unit = str(value.get("unit", "")).strip()
        source = str(value.get("source", "")).strip()
        applies_to = str(value.get("applies_to", "")).strip()
        if not unit or not source or not applies_to:
            raise ValueError(f"evidence for {name} requires unit, source, and applies_to")
        uncertainty = value.get("uncertainty")
        if uncertainty is not None:
            uncertainty = float(uncertainty)
            if not math.isfinite(uncertainty) or uncertainty < 0.0:
                raise ValueError(f"evidence uncertainty for {name} must be finite and non-negative")
        valid_range_raw = value.get("valid_range")
        valid_range = None
        if valid_range_raw is not None:
            if not isinstance(valid_range_raw, (list, tuple)) or len(valid_range_raw) != 2:
                raise ValueError(f"valid_range for {name} must contain two numbers")
            valid_range = (float(valid_range_raw[0]), float(valid_range_raw[1]))
            if not all(math.isfinite(item) for item in valid_range) or valid_range[1] < valid_range[0]:
                raise ValueError(f"valid_range for {name} is invalid")
        return cls(
            value=value.get("value"), unit=unit, source=source,
            uncertainty=uncertainty, applies_to=applies_to,
            valid_range=valid_range, notes=str(value.get("notes", "")).strip(),
        )


@dataclass(frozen=True)
class AssetInputAudit:
    asset_id: str
    component_model: str
    missing_evidence: tuple[str, ...]
    issues: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing_evidence and not self.issues


def _read_mapping(path: str | Path) -> Mapping[str, Any]:
    path_obj = Path(path)
    text = path_obj.read_text(encoding="utf-8")
    if path_obj.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("YAML input requires the optional PyYAML dependency") from exc
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, Mapping):
        raise ValueError("asset input file must contain an object")
    return loaded


def load_evidenced_asset_catalog(path: str | Path) -> dict[str, Any]:
    """Load a private catalog and reject accidental optimisation settings."""

    data = dict(_read_mapping(path))
    if str(data.get("schema_version", "")) not in {"0.1", "0.2"}:
        raise ValueError("unsupported evidenced asset catalog schema")
    if data.get("privacy") != "private":
        raise ValueError("evidenced asset catalog must declare privacy='private'")
    policy = data.get("parameter_policy", {})
    if not isinstance(policy, Mapping) or policy.get("plant_data_optimization") is not False:
        raise ValueError("catalog must explicitly disable plant-data parameter optimization")
    assets = data.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("catalog assets must be a non-empty list")
    return data


def audit_asset_evidence(asset: Mapping[str, Any]) -> AssetInputAudit:
    """Check provenance without changing or filling model parameters."""

    asset_id = str(asset.get("asset_id", "")).strip()
    model = str(asset.get("component_model", "")).strip()
    if not asset_id or not model:
        raise ValueError("asset_id and component_model are required")
    parameters = asset.get("model_parameters", {})
    evidence = asset.get("parameter_evidence", {})
    if not isinstance(parameters, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError(f"{asset_id}: model_parameters and parameter_evidence must be objects")
    missing: list[str] = []
    issues: list[str] = []
    for name, parameter in parameters.items():
        if parameter is None:
            missing.append(str(name))
            continue
        if name not in evidence:
            missing.append(str(name))
            continue
        try:
            item = ParameterEvidence.from_mapping(evidence[name], name=str(name))
        except ValueError as exc:
            issues.append(str(exc))
            continue
        if item.applies_to not in {asset_id, "asset", "common"}:
            issues.append(f"{asset_id}:{name} evidence applies_to={item.applies_to!r}")
        if item.value is not None:
            try:
                if isinstance(parameter, (int, float)) and not math.isclose(float(item.value), float(parameter), rel_tol=1e-12, abs_tol=1e-12):
                    issues.append(f"{asset_id}:{name} evidence value differs from model_parameters")
            except (TypeError, ValueError):
                issues.append(f"{asset_id}:{name} evidence value is not numeric")
    return AssetInputAudit(asset_id, model, tuple(missing), tuple(issues))


def audit_evidenced_asset_catalog(catalog: Mapping[str, Any]) -> tuple[AssetInputAudit, ...]:
    assets = catalog.get("assets")
    if not isinstance(assets, list):
        raise ValueError("catalog assets must be a list")
    return tuple(audit_asset_evidence(asset) for asset in assets)


__all__ = [
    "AssetInputAudit",
    "ParameterEvidence",
    "audit_asset_evidence",
    "audit_evidenced_asset_catalog",
    "load_evidenced_asset_catalog",
]
