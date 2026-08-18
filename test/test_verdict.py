"""The verdict decides whether a run may be used, so it must be hard to fool."""

import unittest

try:
    from robot_parameter_identification import campaign
except ImportError as error:
    raise unittest.SkipTest(f"needs pinocchio: {error}") from error


def joint(residual, validation):
    return {"residual_rms_a": residual, "validation_rms_a": validation}


def result(joints, phases=None, aborted=None):
    outcome = campaign.CampaignResult(
        joints=list(joints),
        phases=phases if phases is not None else [{"peak_current_a": 10.0}],
    )
    outcome.aborted = aborted
    return outcome


class JudgeJointTest(unittest.TestCase):
    def test_validation_tracking_residual_passes(self):
        state, _ = campaign.judge_joint(joint(0.003, 0.004), signal_a=10.0)
        self.assertEqual(state, "pass")

    def test_moderate_divergence_warns(self):
        state, reason = campaign.judge_joint(joint(0.01, 0.05), signal_a=1.0)
        self.assertEqual(state, "warn")
        self.assertIn("provisional", reason)

    def test_large_divergence_fails(self):
        state, reason = campaign.judge_joint(joint(0.003, 1.0), signal_a=10.0)
        self.assertEqual(state, "fail")
        self.assertIn("not generalised", reason)

    def test_missing_validation_is_unknown_not_pass(self):
        state, _ = campaign.judge_joint({"residual_rms_a": 0.001}, signal_a=10.0)
        self.assertEqual(state, "unknown")

    def test_ratio_is_ignored_when_both_numbers_are_noise(self):
        """Chasing a 5x ratio between two numbers at 0.05% of signal is noise."""
        state, reason = campaign.judge_joint(
            joint(0.0005, 0.0025), signal_a=10.0)
        self.assertEqual(state, "pass")
        self.assertIn("largest current", reason)

    def test_negligible_floor_does_not_hide_a_real_failure(self):
        state, _ = campaign.judge_joint(joint(0.003, 1.0), signal_a=10.0)
        self.assertEqual(state, "fail")

    def test_floor_scales_with_the_signal(self):
        small = campaign.judge_joint(joint(0.0005, 0.0025), signal_a=0.1)[0]
        large = campaign.judge_joint(joint(0.0005, 0.0025), signal_a=10.0)[0]
        self.assertEqual(large, "pass")
        self.assertNotEqual(small, "pass")


class VerdictTest(unittest.TestCase):
    def test_all_good_joints_pass(self):
        verdict = result([joint(0.003, 0.004)] * 3).verdict()
        self.assertEqual(verdict["state"], "pass")
        self.assertEqual(verdict["failed_joints"], [])

    def test_one_bad_joint_fails_the_run(self):
        verdict = result(
            [joint(0.003, 0.004), joint(0.003, 2.0), joint(0.003, 0.004)]
        ).verdict()
        self.assertEqual(verdict["state"], "fail")
        self.assertEqual(verdict["failed_joints"], [2])

    def test_a_warning_joint_downgrades_the_run(self):
        verdict = result(
            [joint(0.003, 0.004), joint(0.01, 0.05)],
            phases=[{"peak_current_a": 1.0}],
        ).verdict()
        self.assertEqual(verdict["state"], "warn")

    def test_an_aborted_run_never_passes(self):
        verdict = result([joint(0.003, 0.004)], aborted="operator stop").verdict()
        self.assertEqual(verdict["state"], "fail")

    def test_a_run_with_no_joints_fails(self):
        self.assertEqual(result([]).verdict()["state"], "fail")

    def test_verdict_is_part_of_the_serialised_result(self):
        payload = result([joint(0.003, 0.004)]).as_dict()
        self.assertIn("verdict", payload)
        self.assertEqual(payload["verdict"]["state"], "pass")


if __name__ == "__main__":
    unittest.main()
