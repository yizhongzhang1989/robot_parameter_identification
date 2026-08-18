"""Robot-agnostic dynamic parameter identification.

A new arm is a new YAML profile, not a code change:

    from robot_parameter_identification import RobotProfile, ArmModel, Campaign
    from robot_parameter_identification.plants import SimulatedPlant

    profile = RobotProfile.from_yaml("profiles/my_arm.yaml")
    arm = ArmModel.from_profile(urdf_text, profile)
    result = Campaign(arm, plant, default_plan(profile)).run()
"""

from .profile import RobotProfile, ProfileError, load, available
from .identification import (
    ArmModel, JointRegression, fit_joint, predict_joint, friction_row,
    identifiable_columns, truncated_solve, stacked_condition_number,
    urdf_from_xacro,
)
from .excitation import (
    DesignLimits, StaticPlan, FourierTrajectory, FrictionSweep,
    design_static_poses, design_friction_sweeps, design_fourier_trajectory,
)
from .model import ModelComponents, DEFAULT_COMPONENTS, extra_row
from .consistency import (
    LinkVerdict, pseudo_inertia, check_link, check_parameters, summarise,
)
from .campaign import (
    Campaign, CampaignPlan, CampaignResult, Observation, PhaseReport, Abort,
    Plant, EnvelopeMonitor, PHASES, PHASE_GRAVITY, PHASE_FRICTION,
    PHASE_INERTIA, PHASE_VALIDATION, MAXIMUM_CONDITION,
    campaign_bounds, clamp_campaign_plan, default_plan, judge_joint,
    sweep_speeds, sweep_amplitude_deg,
)

__all__ = [
    "RobotProfile", "ProfileError", "load", "available",
    "ModelComponents", "DEFAULT_COMPONENTS", "extra_row",
    "LinkVerdict", "pseudo_inertia", "check_link", "check_parameters",
    "summarise",
    "ArmModel", "JointRegression", "fit_joint", "predict_joint", "friction_row",
    "identifiable_columns", "truncated_solve", "stacked_condition_number",
    "urdf_from_xacro",
    "DesignLimits", "StaticPlan", "FourierTrajectory", "FrictionSweep",
    "design_static_poses", "design_friction_sweeps", "design_fourier_trajectory",
    "Campaign", "CampaignPlan", "CampaignResult", "Observation", "PhaseReport",
    "Abort", "Plant", "EnvelopeMonitor", "PHASES", "PHASE_GRAVITY",
    "PHASE_FRICTION", "PHASE_INERTIA", "PHASE_VALIDATION", "MAXIMUM_CONDITION",
    "campaign_bounds", "clamp_campaign_plan", "default_plan", "judge_joint",
    "sweep_speeds", "sweep_amplitude_deg",
]
