"""V7.2.4 n8n workflow contract tests."""

import json
from pathlib import Path


WORKFLOW_PATH = Path(__file__).resolve().parents[1] / "n8n" / "v7.2.4-monitoring-contract.json"


def load_workflow():
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_v724_workflow_is_valid_json_and_has_expected_nodes():
    workflow = load_workflow()
    node_names = {node["name"] for node in workflow["nodes"]}

    assert workflow["name"] == "Regula V7.2.4 - Monitoring Contract"
    assert {"Manual Trigger", "Build Monitoring Event", "Call Regula /monitor/analyze"} <= node_names


def test_v724_workflow_calls_monitor_analyze_with_post():
    workflow = load_workflow()
    http_node = next(
        node for node in workflow["nodes"] if node["name"] == "Call Regula /monitor/analyze"
    )

    assert http_node["parameters"]["method"] == "POST"
    assert "/monitor/analyze" in http_node["parameters"]["url"]
    assert http_node["parameters"]["sendBody"] is True
    assert http_node["parameters"]["specifyBody"] == "json"


def test_v724_workflow_uses_the_v71_monitoring_payload_shape():
    workflow = load_workflow()
    event_node = next(
        node for node in workflow["nodes"] if node["name"] == "Build Monitoring Event"
    )
    assignments = event_node["parameters"]["assignments"]["assignments"]
    names = {assignment["name"] for assignment in assignments}

    assert {"regulation", "event"} <= names
    assert "new_regulation" in next(
        assignment["value"] for assignment in assignments if assignment["name"] == "event"
    )
