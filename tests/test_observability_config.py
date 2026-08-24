from __future__ import annotations

import json
from pathlib import Path

import yaml


def test_prometheus_scrapes_api_and_qdrant_over_docker_network() -> None:
    config = yaml.safe_load(
        Path("observability/prometheus/prometheus.yml").read_text(encoding="utf-8")
    )

    jobs = {job["job_name"]: job for job in config["scrape_configs"]}

    assert jobs["rag-steel"]["metrics_path"] == "/metrics"
    assert jobs["rag-steel"]["static_configs"][0]["targets"] == ["api:8005"]
    assert jobs["qdrant"]["metrics_path"] == "/metrics"
    assert jobs["qdrant"]["static_configs"][0]["targets"] == ["qdrant:6333"]


def test_grafana_datasource_points_to_prometheus_service() -> None:
    config = yaml.safe_load(
        Path("observability/grafana/provisioning/datasources/prometheus.yml").read_text(
            encoding="utf-8"
        )
    )

    datasource = config["datasources"][0]
    assert datasource["name"] == "Prometheus"
    assert datasource["uid"] == "prometheus"
    assert datasource["url"] == "http://prometheus:9090"
    assert datasource["editable"] is False


def test_grafana_dashboard_provider_points_to_provisioned_path() -> None:
    config = yaml.safe_load(
        Path("observability/grafana/provisioning/dashboards/dashboards.yml").read_text(
            encoding="utf-8"
        )
    )

    provider = config["providers"][0]
    assert provider["folder"] == "RAG Steel"
    assert provider["options"]["path"] == "/var/lib/grafana/dashboards"
    assert provider["allowUiUpdates"] is False


def test_dashboard_json_is_valid_and_contains_core_panels() -> None:
    dashboard = json.loads(
        Path("observability/grafana/dashboards/rag-steel-overview.json").read_text(
            encoding="utf-8"
        )
    )

    assert dashboard["title"] == "RAG Steel Overview"
    titles = {panel.get("title") for panel in dashboard["panels"] if panel.get("title")}
    assert "Search RPS" in titles
    assert "Successful RPS" in titles
    assert "Total Requests" in titles
    assert "Successful Requests" in titles
    assert "5xx Requests" in titles
    assert "SERVICE_BUSY Count" in titles
    assert "HTTP p95" in titles
    assert "In-flight" in titles
    assert "SERVICE_BUSY Rate" in titles
    assert "DeepSeek p95" in titles
    assert "Embedding p95" in titles
    assert "Qdrant p95" in titles
    assert "Ranking p95" in titles
    assert "API Errors" in titles
    assert "Search Result Status" in titles
    assert "Search Outcome Count" in titles
    assert "Qdrant Collection Dense Vectors" in titles

    in_flight = next(panel for panel in dashboard["panels"] if panel.get("title") == "In-flight")
    assert in_flight["targets"][0]["expr"] == "rag_search_requests_in_flight"


def test_dashboard_count_panels_use_selected_range_increase() -> None:
    dashboard = json.loads(
        Path("observability/grafana/dashboards/rag-steel-overview.json").read_text(
            encoding="utf-8"
        )
    )
    panels = {panel.get("title"): panel for panel in dashboard["panels"] if panel.get("title")}

    expected = {
        "Total Requests": [
            "rag_http_requests_total",
            'path="/v2/search"',
            "or vector(0)",
        ],
        "Successful Requests": [
            "rag_http_requests_total",
            'path="/v2/search"',
            'status_code="200"',
            "or vector(0)",
        ],
        "5xx Requests": [
            "rag_http_requests_total",
            'path="/v2/search"',
            'status_code=~"5.."',
            "or vector(0)",
        ],
        "SERVICE_BUSY Count": [
            "rag_api_errors_total",
            'code="SERVICE_BUSY"',
            "or vector(0)",
        ],
        "Search Outcome Count": [
            "rag_search_requests_total",
            "result_status",
        ],
    }

    for title, required_parts in expected.items():
        panel = panels[title]
        expr = panel["targets"][0]["expr"]
        assert "increase(" in expr
        assert "$__range" in expr
        assert "rate(" not in expr
        for part in required_parts:
            assert part in expr
