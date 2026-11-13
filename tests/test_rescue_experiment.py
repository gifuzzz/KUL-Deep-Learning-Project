import hashlib
import json
import subprocess
import sys

import pytest

from sb3_contrib import MaskablePPO

from agents.cnn import RescueCNN
from rescue_env.envs.single_agent import RescueEnv


@pytest.mark.parametrize("n_envs,steps,rollout_steps", [(1, 8, 8), (8, 1024, 128)])
def test_continuation_preserves_source_and_records_contract(tmp_path, n_envs, steps, rollout_steps):
    env = RescueEnv(max_steps=64, victim_distance=3)
    try:
        model = MaskablePPO("MultiInputPolicy", env, n_steps=8, batch_size=8,
                            n_epochs=1, device="cpu", seed=7,
                            policy_kwargs={"features_extractor_class": RescueCNN,
                                           "normalize_images": False})
        source = tmp_path / "source.zip"
        model.save(source)
        digest = hashlib.sha256(source.read_bytes()).digest()
        output = tmp_path / "experiment"
        command = [sys.executable, "-m", "training.train_rescue_experiment", "--model",
                   str(source), "--output", str(output), "--steps", str(steps), "--device", "cpu",
                   "--n-envs", str(n_envs),
                   "--navigation-shaping"]
        subprocess.run(command, check=True, capture_output=True, text=True)
        assert hashlib.sha256(source.read_bytes()).digest() == digest
        config = json.loads((output / "config.json").read_text())
        assert config["navigation_shaping"]
        assert config["validation_seeds"] == [10000, 10050]
        restored = MaskablePPO.load(output / "last_model", env=env, device="cpu")
        assert restored.num_timesteps == steps
        assert restored.n_steps == rollout_steps
        repeat = subprocess.run(command, capture_output=True, text=True)
        assert repeat.returncode != 0
        assert "new directory" in repeat.stderr
    finally:
        env.close()
