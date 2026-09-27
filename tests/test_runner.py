"""The command line: the data report after a finished run, and Ctrl-C."""

import os
import sys
from types import SimpleNamespace

import pytest

from goes_downloader import runner

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def cli(monkeypatch):
    """cli.run(*args, outcome=...) runs runner.main with the download replaced by `outcome` (an
    exception to raise, or None to finish); cli.reports collects the configs reported on."""
    state = SimpleNamespace(reports=[])
    monkeypatch.setattr(runner, "print_report", lambda cfg, **kw: state.reports.append(cfg))

    def run(*args, outcome=None):
        async def fake_run_all_days(cfg):
            if outcome is not None:
                raise outcome

        monkeypatch.setattr(runner, "run_all_days", fake_run_all_days)
        monkeypatch.setattr(sys, "argv", ["download_raw.py", "--config", os.path.join(REPO, "config.json"), *args])
        runner.main()

    state.run = run
    return state


def test_report_follows_a_finished_run(cli, capsys):
    cli.run()
    assert len(cli.reports) == 1 and cli.reports[0].start_date_utc
    assert "Data report for" in capsys.readouterr().out


def test_no_report_option(cli):
    cli.run("--no-report")
    assert cli.reports == []


def test_ctrl_c_exits_without_a_report(cli, capsys):
    with pytest.raises(SystemExit) as stop:
        cli.run(outcome=KeyboardInterrupt())
    assert stop.value.code == 130
    assert "Interrupted" in capsys.readouterr().err
    assert cli.reports == []
