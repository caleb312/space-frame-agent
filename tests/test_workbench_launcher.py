"""Launcher failure/reuse paths; no server shutdowns or package downloads."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import launch_workbench as launcher


class WorkbenchLauncherTests(unittest.TestCase):
    def test_existing_workbench_opens_without_preparing_environment(self):
        with patch('sys.argv', ['launcher']), patch.object(launcher, 'service_state', return_value='ours'), \
                patch.object(launcher.webbrowser, 'open') as browser, \
                patch.object(launcher, 'prepare_runtime') as prepare:
            self.assertEqual(launcher.main(), 0)
            browser.assert_called_once_with('http://127.0.0.1:8765/')
            prepare.assert_not_called()

    def test_other_application_is_not_opened_or_replaced(self):
        with patch('sys.argv', ['launcher']), patch.object(launcher, 'service_state', return_value='other'), \
                patch.object(launcher.webbrowser, 'open') as browser, \
                patch.object(launcher, 'prepare_runtime') as prepare:
            self.assertEqual(launcher.main(), 1)
            browser.assert_not_called()
            prepare.assert_not_called()

    def test_environment_failure_does_not_start_application(self):
        with patch('sys.argv', ['launcher']), patch.object(launcher, 'service_state', return_value='unavailable'), \
                patch.object(launcher, 'prepare_runtime', side_effect=RuntimeError('准备失败')), \
                patch.object(launcher.subprocess, 'call') as application:
            self.assertEqual(launcher.main(), 1)
            application.assert_not_called()

    def test_ready_environment_never_calls_package_installer(self):
        launcher.LOCAL.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=launcher.LOCAL, prefix='launcher-test-') as folder:
            root = Path(folder)
            env = root / 'env'
            python = env / ('Scripts/python.exe' if launcher.os.name == 'nt' else 'bin/python')
            python.parent.mkdir(parents=True)
            python.touch()
            (root / 'requirements.txt').write_text('numpy>=2.0,<3\n', encoding='utf-8')
            with patch.object(launcher, 'ROOT', root), patch.object(launcher, 'ENV', env), \
                    patch.object(launcher, 'LOCAL', root / 'local'), \
                    patch.object(launcher, 'dependency_check', return_value=True), \
                    patch.object(launcher, 'run_logged') as installer, \
                    patch.object(launcher.venv, 'EnvBuilder') as builder:
                self.assertEqual(launcher.prepare_runtime(), python)
                installer.assert_not_called()
                builder.assert_not_called()

    def test_start_uses_prepared_python_and_requested_port(self):
        with patch('sys.argv', ['launcher', '--port', '8766', '--provider', 'qwen']), \
                patch.object(launcher, 'service_state', return_value='unavailable'), \
                patch.object(launcher, 'prepare_runtime', return_value=Path('prepared/python.exe')), \
                patch.object(launcher.subprocess, 'call', return_value=0) as application:
            self.assertEqual(launcher.main(), 0)
            application.assert_called_once_with([
                str(Path('prepared/python.exe')), str(launcher.ROOT / 'web_main.py'), '--open',
                '--port', '8766', '--provider', 'qwen'], cwd=launcher.ROOT)


if __name__ == '__main__':
    unittest.main()
