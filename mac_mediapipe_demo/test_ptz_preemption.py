"""Test PTZ command preemption, rapid overwrite, and immediate stop logic.

Tests:
1. PTZ moves right (pan=+1.0), interrupted halfway by send_stop().
2. PTZ moves left (pan=-1.0), interrupted halfway by send_stop().
3. Rapid sequence of move commands: verifies newer commands overwrite older ones.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock

from ptz_worker import PTZCommandWorker


class TestPTZCommandPreemption(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_ptz = MagicMock()
        self.mock_ptz.is_moving = False
        self.worker = PTZCommandWorker(self.mock_ptz, min_interval=0.01)
        self.worker.start()

    def tearDown(self) -> None:
        self.worker.stop_and_join()

    def test_move_right_and_stop_halfway(self) -> None:
        """1. Move to far right (+1.0), stop halfway through."""
        print("\n[Test 1] Moving far right (+1.0)...")
        self.worker.send_move(1.0, 0.0, 1.0)
        time.sleep(0.10)  # Allow move to execute and run halfway

        print("[Test 1] Interrupting halfway with send_stop()...")
        self.worker.send_stop(force=True)
        time.sleep(0.05)

        # Verify move(1.0, 0.0, 1.0) was called, followed by stop(force=True)
        self.mock_ptz.move.assert_called_with(1.0, 0.0, 1.0)
        self.mock_ptz.stop.assert_called_with(force=True)
        print(" -> Test 1 PASSED: Move right was interrupted halfway by stop.")

    def test_move_left_and_stop_halfway(self) -> None:
        """2. Move to far left (-1.0), stop halfway through."""
        print("\n[Test 2] Moving far left (-1.0)...")
        self.worker.send_move(-1.0, 0.0, 1.0)
        time.sleep(0.10)  # Allow move to execute and run halfway

        print("[Test 2] Interrupting halfway with send_stop()...")
        self.worker.send_stop(force=True)
        time.sleep(0.05)

        # Verify move(-1.0, 0.0, 1.0) was called, followed by stop(force=True)
        self.mock_ptz.move.assert_called_with(-1.0, 0.0, 1.0)
        self.mock_ptz.stop.assert_called_with(force=True)
        print(" -> Test 2 PASSED: Move left was interrupted halfway by stop.")

    def test_new_command_overwrites_old_command(self) -> None:
        """3. Rapid commands overwrite obsolete intermediate commands."""
        print("\n[Test 3] Rapidly queuing obsolete commands (0.1, 0.3, 0.5, 0.8 -> -1.0)...")

        # Simulate camera SOAP HTTP move request taking 0.08s
        def slow_move(pan, tilt, timeout):
            time.sleep(0.08)

        self.mock_ptz.move.side_effect = slow_move

        # Send first command (will start executing)
        self.worker.send_move(0.1, 0.0, 1.0)
        time.sleep(0.01)

        # While 0.1 is executing in background, send obsolete intermediate commands
        self.worker.send_move(0.3, 0.0, 1.0)
        self.worker.send_move(0.5, 0.0, 1.0)
        self.worker.send_move(0.8, 0.0, 1.0)
        # Latest command (move far left -1.0)
        self.worker.send_move(-1.0, 0.0, 1.0)

        # Wait for worker to finish execution loop
        time.sleep(0.20)

        # Extract all pan values passed to mock_ptz.move
        calls = self.mock_ptz.move.call_args_list
        pan_calls = [call[0][0] for call in calls]
        print(f"[Test 3] Actual move commands executed by PTZ: {pan_calls}")

        # Intermediate commands 0.3, 0.5, 0.8 must be skipped/overwritten!
        self.assertNotIn(0.3, pan_calls, "Intermediate command 0.3 should have been overwritten!")
        self.assertNotIn(0.5, pan_calls, "Intermediate command 0.5 should have been overwritten!")
        self.assertNotIn(0.8, pan_calls, "Intermediate command 0.8 should have been overwritten!")
        self.assertEqual(pan_calls[-1], -1.0, "Latest command (-1.0) must be executed!")
        print(" -> Test 3 PASSED: Newer commands successfully overwrote old intermediate commands.")


if __name__ == "__main__":
    unittest.main()
