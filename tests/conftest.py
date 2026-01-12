"""Pytest configuration for hill_mixture_mmm tests."""

from __future__ import annotations

import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest

FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"
BENCHMARK_ARTIFACTS_DIR = FIGURES_DIR
BENCHMARK_LOG_DIR = Path(__file__).parent.parent / "results" / "test_logs"


class BenchmarkSessionReporter:
    """Persist a compact text report for benchmark-oriented pytest runs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.started_at = datetime.now()
        self.started_perf = perf_counter()
        self.is_benchmark_session = False
        self.nodeids: list[str] = []
        self.reports: list[dict[str, Any]] = []
        self.warning_lines: list[str] = []
        self.output_path: Path | None = None

    def pytest_collection_modifyitems(
        self,
        session: pytest.Session,
        config: pytest.Config,
        items: list[pytest.Item],
    ) -> None:
        """Enable file reporting only when a benchmark test was collected."""
        self.is_benchmark_session = any(
            item.get_closest_marker("benchmark_smoke") or item.get_closest_marker("benchmark_full")
            for item in items
        )
        if not self.is_benchmark_session:
            return

        self.nodeids = [item.nodeid for item in items]
        self.root.mkdir(parents=True, exist_ok=True)
        timestamp = self.started_at.strftime("%Y%m%d_%H%M%S")
        marker_suffix = "benchmark"
        selected_expr = getattr(config.option, "markexpr", "") or ""
        if "benchmark_full" in selected_expr:
            marker_suffix = "benchmark_full"
        elif "benchmark_smoke" in selected_expr:
            marker_suffix = "benchmark_smoke"
        self.output_path = self.root / f"pytest_{marker_suffix}_{timestamp}.log"

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Store outcomes and failure bodies for the final text report."""
        if not self.is_benchmark_session:
            return
        if report.when != "call":
            if report.failed or report.skipped:
                self.reports.append(
                    {
                        "nodeid": report.nodeid,
                        "when": report.when,
                        "outcome": report.outcome,
                        "duration": report.duration,
                        "details": report.longreprtext,
                    }
                )
            return

        self.reports.append(
            {
                "nodeid": report.nodeid,
                "when": report.when,
                "outcome": report.outcome,
                "duration": report.duration,
                "details": report.longreprtext if report.failed else None,
            }
        )

    def pytest_warning_recorded(
        self,
        warning_message: warnings.WarningMessage,
        when: str,
        nodeid: str | None,
        location: tuple[str, int, str] | None,
    ) -> None:
        """Keep benchmark warnings in the persisted report."""
        if not self.is_benchmark_session:
            return
        node = nodeid or "<session>"
        warning_type = warning_message.category.__name__
        message = str(warning_message.message)
        self.warning_lines.append(f"{when} {node} [{warning_type}] {message}")

    def pytest_terminal_summary(
        self,
        terminalreporter: Any,
        exitstatus: int,
        config: pytest.Config,
    ) -> None:
        """Write the benchmark report at the end of the session."""
        if not self.is_benchmark_session or self.output_path is None:
            return

        duration = perf_counter() - self.started_perf
        counts = Counter(report["outcome"] for report in self.reports)
        selected_args = " ".join(str(arg) for arg in config.invocation_params.args)
        lines = [
            "Pytest Benchmark Report",
            f"Started: {self.started_at.isoformat(timespec='seconds')}",
            f"DurationSeconds: {duration:.2f}",
            f"ExitStatus: {exitstatus}",
            f"CommandArgs: {selected_args}",
            f"CollectedItems: {len(self.nodeids)}",
            f"Passed: {counts.get('passed', 0)}",
            f"Failed: {counts.get('failed', 0)}",
            f"Skipped: {counts.get('skipped', 0)}",
            "",
            "Case Results:",
        ]

        for report in self.reports:
            lines.append(
                f"- {report['nodeid']} :: {report['outcome']} :: "
                f"{report['duration']:.2f}s [{report['when']}]"
            )
            if report["details"]:
                lines.append(report["details"].rstrip())
                lines.append("")

        if self.warning_lines:
            lines.extend(["Warnings:"])
            lines.extend(self.warning_lines)

        self.output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        terminalreporter.write_line(f"Benchmark report written to {self.output_path}")


def pytest_configure(config):
    """Add custom markers and register benchmark log persistence."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m not slow')")
    config.addinivalue_line("markers", "benchmark_smoke: quick benchmark smoke tests")
    config.addinivalue_line("markers", "benchmark_full: full multi-seed benchmark tests")
    config.pluginmanager.register(
        BenchmarkSessionReporter(BENCHMARK_LOG_DIR), "benchmark-session-reporter"
    )


@pytest.fixture
def output_dir() -> Path:
    """Return the paper figures directory for saving visualization outputs."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


@pytest.fixture
def benchmark_output_root() -> Path:
    """Return the paper-submodule figure directory for benchmark test visualizations."""
    BENCHMARK_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return BENCHMARK_ARTIFACTS_DIR
