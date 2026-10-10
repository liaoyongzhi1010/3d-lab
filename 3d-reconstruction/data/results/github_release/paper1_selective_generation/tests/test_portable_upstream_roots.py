import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


class PortableUpstreamRootsTest(unittest.TestCase):
    def run_script(self, script, *args):
        env = os.environ.copy()
        env.pop("GEN3R_ROOT", None)
        env.pop("VGGT_ROOT", None)
        return subprocess.run(
            [sys.executable, str(SCRIPTS / script), *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_help_works_without_upstream_repositories(self):
        for script in ("e012_vesg.py", "precompute_visibility.py"):
            with self.subTest(script=script):
                result = self.run_script(script, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--gen3r_root", result.stdout)
                self.assertIn("--vggt_root", result.stdout)

    def assert_setup_error(self, result):
        self.assertNotEqual(result.returncode, 0)
        error = result.stdout + result.stderr
        self.assertIn("RuntimeError", error)
        self.assertIn("https://github.com/JaceyHuang/Gen3R", error)
        self.assertIn("https://github.com/facebookresearch/vggt", error)
        return error

    def test_missing_roots_report_official_setup_guidance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_gen3r = str(Path(temp_dir) / "missing-gen3r")
            missing_vggt = str(Path(temp_dir) / "missing-vggt")
            for script in ("e012_vesg.py", "precompute_visibility.py"):
                with self.subTest(script=script):
                    result = self.run_script(
                        script,
                        "--gen3r_root",
                        missing_gen3r,
                        "--vggt_root",
                        missing_vggt,
                    )
                    error = self.assert_setup_error(result)
                    self.assertIn("GEN3R_ROOT", error)
                    self.assertIn("VGGT_ROOT", error)

    def test_missing_upstream_modules_report_official_setup_guidance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            gen3r_root = Path(temp_dir) / "Gen3R"
            vggt_root = Path(temp_dir) / "vggt"
            gen3r_root.mkdir()
            vggt_root.mkdir()
            for script in ("e012_vesg.py", "precompute_visibility.py"):
                with self.subTest(script=script):
                    result = self.run_script(
                        script,
                        "--gen3r_root",
                        str(gen3r_root),
                        "--vggt_root",
                        str(vggt_root),
                    )
                    self.assert_setup_error(result)


if __name__ == "__main__":
    unittest.main()
