from __future__ import annotations

from pathlib import Path

import yaml


def _load_compose() -> dict[str, object]:
    return yaml.safe_load(Path("compose.yaml").read_text(encoding="utf-8"))


def test_api_healthcheck_uses_readiness_endpoint() -> None:
    compose = _load_compose()
    api = compose["services"]["api"]
    healthcheck = api["healthcheck"]
    command = healthcheck["test"]

    assert "/app/.venv/bin/python" in command
    assert "/health/ready" in command[-1]
    assert "/health/live" not in command[-1]


def test_indexer_uses_all_source_csvs_without_alias_switch() -> None:
    compose = _load_compose()
    indexer = compose["services"]["indexer"]

    volumes = indexer["volumes"]
    assert "./mapping_results.csv:/data/mapping_results.csv:ro" in volumes
    assert "./butterfly_mapping_results.csv:/data/butterfly_mapping_results.csv:ro" in volumes
    assert "./competitor_ld_mapping.csv:/data/competitor_ld_mapping.csv:ro" in volumes

    command = indexer["command"]
    assert command[:2] == ["/app/.venv/bin/python", "indexer.py"]
    assert command.count("--csv") == 3
    assert "/data/mapping_results.csv" in command
    assert "/data/butterfly_mapping_results.csv" in command
    assert "/data/competitor_ld_mapping.csv" in command
    assert "--recreate" not in command


def test_dockerfile_uses_pinned_runtime_and_non_root_execution() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.11.13-slim" in dockerfile
    assert "ghcr.io/astral-sh/uv:0.11.14" in dockerfile
    assert 'CMD ["/app/.venv/bin/uvicorn"' in dockerfile
    assert "CMD [\"uv\", \"run\"" not in dockerfile
    assert "USER app" in dockerfile


def test_dockerignore_excludes_secrets_and_source_csvs() -> None:
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

    assert ".env" in dockerignore
    assert "mapping_results.csv" in dockerignore
    assert "butterfly_mapping_results.csv" in dockerignore
    assert "competitor_ld_mapping.csv" in dockerignore


def test_observability_services_use_localhost_ports_and_pinned_images() -> None:
    compose = _load_compose()

    prometheus = compose["services"]["prometheus"]
    grafana = compose["services"]["grafana"]

    assert prometheus["image"] == "prom/prometheus:v3.13.2"
    assert grafana["image"] == "grafana/grafana:12.4.8"
    assert "latest" not in prometheus["image"]
    assert "latest" not in grafana["image"]
    assert prometheus["ports"] == ["127.0.0.1:9095:9090"]
    assert grafana["ports"] == ["127.0.0.1:3005:3000"]
    assert grafana["environment"]["GF_SECURITY_ADMIN_USER"] == "${GRAFANA_ADMIN_USER:-admin}"
    assert (
        grafana["environment"]["GF_SECURITY_ADMIN_PASSWORD"]
        == "${GRAFANA_ADMIN_PASSWORD:?GRAFANA_ADMIN_PASSWORD must be set}"
    )


def test_observability_mounts_are_read_only_and_named_volumes_exist() -> None:
    compose = _load_compose()

    prometheus = compose["services"]["prometheus"]
    grafana = compose["services"]["grafana"]

    assert (
        "./observability/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro"
        in prometheus["volumes"]
    )
    assert "./observability/grafana/provisioning:/etc/grafana/provisioning:ro" in grafana["volumes"]
    assert "./observability/grafana/dashboards:/var/lib/grafana/dashboards:ro" in grafana["volumes"]
    assert "prometheus_data:/prometheus" in prometheus["volumes"]
    assert "grafana_data:/var/lib/grafana" in grafana["volumes"]
    assert "prometheus_data" in compose["volumes"]
    assert "grafana_data" in compose["volumes"]


def test_api_does_not_depend_on_observability_services() -> None:
    compose = _load_compose()
    api = compose["services"]["api"]

    depends_on = api.get("depends_on", [])
    if isinstance(depends_on, dict):
        keys = set(depends_on)
    else:
        keys = set(depends_on)

    assert "prometheus" not in keys
    assert "grafana" not in keys
