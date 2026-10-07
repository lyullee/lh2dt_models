"""Serializable component graph contract for visual network editors.

The module is the model-side half of an Aspen/HYSYS-like editor. It stores
node positions, ports, component parameters, and links without depending on a
GUI toolkit. A desktop or web editor can render these records as icons and
later hand the same draft to a network-specific builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class AuthoringPort:
    port_id: str
    phase: str = "mixed"
    direction: str = "bidirectional"

    def __post_init__(self) -> None:
        if not str(self.port_id).strip():
            raise ValueError("port_id must be non-empty")
        if self.phase not in {"liquid", "vapor", "mixed", "thermal", "signal"}:
            raise ValueError("unsupported port phase")
        if self.direction not in {"in", "out", "bidirectional"}:
            raise ValueError("unsupported port direction")


@dataclass(frozen=True)
class AuthoringNode:
    node_id: str
    component_model: str
    label: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    ports: tuple[AuthoringPort, ...] = ()
    position_x: float = 0.0
    position_y: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.node_id).strip() or not str(self.component_model).strip():
            raise ValueError("node_id and component_model must be non-empty")
        if not str(self.label).strip():
            raise ValueError("node label must be non-empty")
        if len({port.port_id for port in self.ports}) != len(self.ports):
            raise ValueError(f"node {self.node_id} has duplicate ports")

    @property
    def port_ids(self) -> frozenset[str]:
        return frozenset(port.port_id for port in self.ports)


@dataclass(frozen=True)
class AuthoringLink:
    link_id: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str
    link_model: str = "pipe"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(str(value).strip() for value in (self.link_id, self.source_node, self.source_port, self.target_node, self.target_port)):
            raise ValueError("link identifiers and ports must be non-empty")
        if self.source_node == self.target_node:
            raise ValueError("self-links are not allowed in an authoring draft")


@dataclass(frozen=True)
class DraftIssue:
    severity: str
    location: str
    message: str


@dataclass(frozen=True)
class NetworkDraft:
    schema_version: str
    nodes: tuple[AuthoringNode, ...]
    links: tuple[AuthoringLink, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NetworkDraft":
        if str(value.get("schema_version", "")) != "0.1":
            raise ValueError("unsupported network draft schema")
        nodes: list[AuthoringNode] = []
        for raw in value.get("nodes", []):
            ports = tuple(AuthoringPort(**port) for port in raw.get("ports", []))
            nodes.append(AuthoringNode(
                node_id=str(raw["node_id"]), component_model=str(raw["component_model"]),
                label=str(raw.get("label", raw["node_id"])),
                parameters=dict(raw.get("parameters", {})), ports=ports,
                position_x=float(raw.get("position_x", 0.0)),
                position_y=float(raw.get("position_y", 0.0)),
                metadata=dict(raw.get("metadata", {})),
            ))
        links = tuple(AuthoringLink(
            link_id=str(raw["link_id"]), source_node=str(raw["source_node"]),
            source_port=str(raw["source_port"]), target_node=str(raw["target_node"]),
            target_port=str(raw["target_port"]), link_model=str(raw.get("link_model", "pipe")),
            parameters=dict(raw.get("parameters", {})),
        ) for raw in value.get("links", []))
        return cls("0.1", tuple(nodes), links, dict(value.get("metadata", {})))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
            "nodes": [
                {
                    "node_id": node.node_id, "component_model": node.component_model,
                    "label": node.label, "parameters": dict(node.parameters),
                    "ports": [
                        {"port_id": port.port_id, "phase": port.phase, "direction": port.direction}
                        for port in node.ports
                    ],
                    "position_x": node.position_x, "position_y": node.position_y,
                    "metadata": dict(node.metadata),
                }
                for node in self.nodes
            ],
            "links": [
                {
                    "link_id": link.link_id, "source_node": link.source_node,
                    "source_port": link.source_port, "target_node": link.target_node,
                    "target_port": link.target_port, "link_model": link.link_model,
                    "parameters": dict(link.parameters),
                }
                for link in self.links
            ],
        }

    def validate(self) -> tuple[DraftIssue, ...]:
        issues: list[DraftIssue] = []
        nodes = {node.node_id: node for node in self.nodes}
        if len(nodes) != len(self.nodes):
            issues.append(DraftIssue("error", "nodes", "node_id values must be unique"))
        link_ids: set[str] = set()
        for link in self.links:
            if link.link_id in link_ids:
                issues.append(DraftIssue("error", f"links.{link.link_id}", "link_id values must be unique"))
            link_ids.add(link.link_id)
            source = nodes.get(link.source_node)
            target = nodes.get(link.target_node)
            if source is None:
                issues.append(DraftIssue("error", link.link_id, f"unknown source node {link.source_node}"))
            elif link.source_port not in source.port_ids:
                issues.append(DraftIssue("error", link.link_id, f"unknown source port {link.source_node}.{link.source_port}"))
            if target is None:
                issues.append(DraftIssue("error", link.link_id, f"unknown target node {link.target_node}"))
            elif link.target_port not in target.port_ids:
                issues.append(DraftIssue("error", link.link_id, f"unknown target port {link.target_node}.{link.target_port}"))
        return tuple(issues)

    @property
    def ready(self) -> bool:
        return not any(issue.severity == "error" for issue in self.validate())


def load_network_draft(path: str | Path) -> NetworkDraft:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("network draft must contain an object")
    return NetworkDraft.from_mapping(value)


def save_network_draft(draft: NetworkDraft, path: str | Path) -> None:
    issues = draft.validate()
    if any(issue.severity == "error" for issue in issues):
        raise ValueError("cannot save invalid network draft: " + "; ".join(issue.message for issue in issues))
    Path(path).write_text(json.dumps(draft.to_mapping(), ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "AuthoringLink", "AuthoringNode", "AuthoringPort", "DraftIssue",
    "NetworkDraft", "load_network_draft", "save_network_draft",
]
