import json

import pytest

from lh2dt import audit_evidenced_asset_catalog, load_evidenced_asset_catalog


def _catalog():
    return {
        "schema_version": "0.2",
        "privacy": "private",
        "parameter_policy": {"plant_data_optimization": False},
        "assets": [{
            "asset_id": "TK-1101",
            "component_model": "homogeneous_tank",
            "model_parameters": {"volume_m3": 10.0, "ambient_UA_W_K": 2.0},
            "parameter_evidence": {
                "volume_m3": {"value": 10.0, "unit": "m3", "source": "GA drawing", "uncertainty": 0.01, "applies_to": "TK-1101"},
                "ambient_UA_W_K": {"value": 2.0, "unit": "W/K", "source": "insulation schedule", "uncertainty": 0.2, "applies_to": "TK-1101"},
            },
        }],
    }


def test_private_catalog_requires_evidence_for_each_value(tmp_path):
    path = tmp_path / "private.json"
    path.write_text(json.dumps(_catalog()), encoding="utf-8")
    catalog = load_evidenced_asset_catalog(path)
    audits = audit_evidenced_asset_catalog(catalog)
    assert audits[0].ready


def test_evidence_mismatch_is_visible(tmp_path):
    catalog = _catalog()
    catalog["assets"][0]["parameter_evidence"]["volume_m3"]["value"] = 11.0
    path = tmp_path / "private.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    audit = audit_evidenced_asset_catalog(load_evidenced_asset_catalog(path))[0]
    assert not audit.ready
    assert any("differs" in issue for issue in audit.issues)


def test_public_or_optimising_catalog_is_rejected(tmp_path):
    catalog = _catalog()
    catalog["privacy"] = "public"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(ValueError, match="privacy"):
        load_evidenced_asset_catalog(path)

