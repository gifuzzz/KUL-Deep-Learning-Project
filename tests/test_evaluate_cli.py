"""A random baseline should run without a checkpoint and preserve scenario seeds."""
import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize("marker", [False, True])
def test_random_evaluation(tmp_path, marker):
    output = tmp_path / "random.json"
    command = [sys.executable, "-m", "training.evaluate", "--random", "--model",
               "missing.zip", "--episodes", "2", "--scenario-start", "11000",
               "--device", "cpu", "--output", str(output)]
    if marker:
        command.append("--memory-agent-marker")
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result["model"] is None
    assert result["memory_agent_marker"] is marker
    assert result["actions"] == [0, 1, 2, 3, 4, 10, 5]
    assert [episode["seed"] for episode in result["episodes"]] == [11000, 11001]
    assert all(0 < episode["steps"] <= 64 for episode in result["episodes"])
    assert all(1 <= episode["unique_positions"] <= episode["steps"]
               and 0 <= episode["outside_steps"] <= episode["steps"]
               for episode in result["episodes"])
    repeat = subprocess.run(command, capture_output=True, text=True)
    assert repeat.returncode != 0
    assert "output already exists" in repeat.stderr
