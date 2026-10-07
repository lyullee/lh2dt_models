import pytest

from lh2dt import AuthoringLink, AuthoringNode, AuthoringPort, NetworkDraft


def _draft():
    return NetworkDraft(
        "0.1",
        nodes=(
            AuthoringNode("tank", "layered_tank", "Tank", ports=(AuthoringPort("vapor"),)),
            AuthoringNode("vent", "vent_stack", "Vent", ports=(AuthoringPort("inlet"),)),
        ),
        links=(AuthoringLink("line", "tank", "vapor", "vent", "inlet"),),
    )


def test_network_draft_roundtrip_mapping_is_ready():
    draft = _draft()
    assert draft.ready
    restored = NetworkDraft.from_mapping(draft.to_mapping())
    assert restored.to_mapping() == draft.to_mapping()


def test_network_draft_reports_unknown_ports():
    draft = NetworkDraft(
        "0.1",
        nodes=(AuthoringNode("tank", "layered_tank", "Tank", ports=(AuthoringPort("vapor"),)),),
        links=(AuthoringLink("line", "tank", "unknown", "other", "inlet"),),
    )
    assert not draft.ready
    assert any("unknown source port" in issue.message for issue in draft.validate())
    assert any("unknown target node" in issue.message for issue in draft.validate())


def test_authoring_port_rejects_invalid_direction():
    with pytest.raises(ValueError):
        AuthoringPort("p", direction="sideways")
