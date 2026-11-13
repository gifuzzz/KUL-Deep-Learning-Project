import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from agents.cnn import RescueCNN
from training.web_viewer import Inspector, export_replay, make_handler


class InspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.viewer = Inspector()
        self.addCleanup(self.viewer.close)

    def policy(self):
        return MaskablePPO(
            "MultiInputPolicy", self.viewer.env, n_steps=4, batch_size=4,
            seed=0, device="cpu",
            policy_kwargs={"features_extractor_class": RescueCNN, "normalize_images": False},
        )

    def test_manual_transition_records_actual_inputs_and_native_outputs(self):
        before = self.viewer.state()
        self.assertEqual(len(before["layers"]["policy"]), 10)
        self.assertEqual(len(before["layers"]["native"]), 15)
        for action in (2, 5, 4, 4):
            after = self.viewer.step(action)
            transition = after["transition"]
            self.assertEqual(transition["input"], before["observation"])
            self.assertEqual(transition["native_input"], before["native_observation"])
            self.assertEqual(transition["native_command"], self.viewer.env.ACTIONS[action])
            self.assertAlmostEqual(transition["native_reward"] + transition["shaping_adjustment"], transition["reward"])
            self.assertEqual(transition["native_output"]["actions_succeeded"], 1)
            json.dumps(after, allow_nan=False)
            before = after
        self.assertTrue(after["finished"])
        self.assertTrue(after["info"]["success"])
        with self.assertRaises(ValueError):
            self.viewer.step(0)

    def test_masked_actions_require_explicit_test_and_report_no_effect(self):
        self.assertFalse(self.viewer.env.action_masks()[5])
        with self.assertRaises(ValueError):
            self.viewer.step(5)
        self.assertEqual(self.viewer.env.steps, 0)
        result = self.viewer.step(5, allow_invalid=True)
        self.assertFalse(result["transition"]["was_valid"])
        self.assertEqual(result["transition"]["native_output"]["actions_succeeded"], 0)
        self.assertEqual(result["scene"]["agents"][0]["r"], 13)

    def test_probabilities_and_value_match_the_real_policy(self):
        self.viewer.model = self.policy()
        state = self.viewer.state()
        policy = self.viewer.model.policy
        tensors, _ = policy.obs_to_tensor(self.viewer.observation)
        with torch.no_grad():
            probabilities = policy.get_distribution(
                tensors, action_masks=np.asarray(state["action_masks"]),
            ).distribution.probs[0].numpy()
            value = float(policy.predict_values(tensors)[0, 0])
        np.testing.assert_allclose(state["policy"]["probabilities"], probabilities)
        self.assertAlmostEqual(state["policy"]["value"], value)
        self.assertEqual(len(state["policy"]["features"]), 128)
        self.assertAlmostEqual(sum(state["policy"]["probabilities"]), 1, places=6)
        for allowed, probability in zip(state["action_masks"], probabilities):
            if not allowed:
                self.assertEqual(probability, 0)
        after = self.viewer.step()
        self.assertEqual(after["transition"]["action"], state["policy"]["suggested_action"])

    def test_reset_replays_sampled_actions_and_releases_old_transition(self):
        self.viewer.model = self.policy()
        first = self.viewer.step(mode="sampled")
        self.viewer.reset(42)
        self.assertIsNone(self.viewer.transition)
        self.assertEqual(self.viewer.total_reward, 0)
        second = self.viewer.step(mode="sampled")
        self.assertEqual(first["transition"]["action"], second["transition"]["action"])
        self.assertEqual(first["observation"], second["observation"])

    def test_checkpoint_loading_and_self_contained_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "models").mkdir()
            self.policy().save(root / "models" / "policy.zip")
            with patch("training.web_viewer.ROOT", root):
                self.viewer.load("models/policy.zip")
                self.assertEqual(self.viewer.checkpoints(), ["models/policy.zip"])
                with self.assertRaises(ValueError):
                    self.viewer.load("../elsewhere.zip")
                self.viewer.env.max_steps = 2
                path = export_replay(self.viewer, root / "replay.html")
                html = path.read_text()
                self.assertNotIn('src="/app.js"', html)
                self.assertNotIn('href="/style.css"', html)
                frames = json.loads(html.split("window.INSPECTION_REPLAY=", 1)[1].split(";</script>", 1)[0])
                self.assertEqual(len(frames), 3)
                self.assertTrue(frames[-1]["finished"])
                self.assertTrue(frames[-1]["transition"]["truncated"])

    def test_reset_validation_preserves_the_episode(self):
        for seed in (-1, True, 1.2, "42", 2**31):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.viewer.reset(seed)
        for action in (-1, 7, True, "1"):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.viewer.step(action)
        self.assertEqual(self.viewer.env.steps, 0)

    def http(self, method, path, data=None, origin=None, host="127.0.0.1:8765"):
        body = json.dumps(data).encode() if data is not None else b""
        headers = f"Host: {host}\r\nConnection: close\r\n"
        if data is not None:
            headers += f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
        if origin:
            headers += f"Origin: {origin}\r\n"
        request = f"{method} {path} HTTP/1.1\r\n{headers}\r\n".encode() + body

        class Connection:
            response = b""

            def makefile(self, *args):
                return io.BytesIO(request)

            def sendall(self, value):
                self.response += bytes(value)

        connection = Connection()
        make_handler(self.viewer)(connection, ("127.0.0.1", 1234), SimpleNamespace(server_port=8765))
        header, body = connection.response.split(b"\r\n\r\n", 1)
        return int(header.split(b" ")[1]), body

    def test_http_serves_assets_and_steps_but_rejects_other_origins(self):
        status, body = self.http("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Observation & action inspector", body)
        status, body = self.http("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["step"], 0)
        status, body = self.http("POST", "/api/step", {"action": 3})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["step"], 1)
        self.assertEqual(self.http("POST", "/api/step", {"action": 0}, origin="https://example.com")[0], 403)
        self.assertEqual(self.http("GET", "/api/state", host="example.com:8765")[0], 403)
        self.assertEqual(self.http("GET", "/../web_viewer.py")[0], 404)
        self.assertEqual(self.http("POST", "/api/reset", [42])[0], 400)
        self.assertEqual(self.viewer.env.steps, 1)


if __name__ == "__main__":
    unittest.main()
