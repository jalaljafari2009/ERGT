"""Gate decision utility for ERGT Phase 3 -> Phase 4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_utils import load_json, save_json  # noqa: E402

GateStatus = Literal["pass", "conditional_pass", "fail"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide ERGT Gate 1 from Phase 3 reports.")
    parser.add_argument(
        "--comparison",
        default="runs/phase3_geo_attention/comparison_results.json",
        help="Path to comparison_results.json.",
    )
    parser.add_argument(
        "--ablation",
        default="runs/phase3_geo_attention/ablation_report.json",
        help="Path to ablation_report.json.",
    )
    parser.add_argument(
        "--output",
        default="runs/gates/phase3_to_phase4_decision.json",
        help="Output gate decision JSON path.",
    )
    parser.add_argument(
        "--confirm-seed",
        default=None,
        help="Optional path to confirm_seed_results.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = load_json(args.comparison)
    ablation = load_json(args.ablation)
    confirm_seed = load_json(args.confirm_seed) if args.confirm_seed else None
    decision = decide_gate(comparison, ablation, confirm_seed=confirm_seed)
    save_json(args.output, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


def decide_gate(
    comparison: dict[str, Any],
    ablation: dict[str, Any],
    *,
    confirm_seed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks = ablation.get("checks", {})
    deltas = ablation.get("deltas", comparison.get("deltas", {}))
    conditions = comparison.get("conditions", {})

    evidence: list[str] = []
    risks: list[str] = []
    required_next_actions: list[str] = []

    if checks.get("all_required_conditions_present"):
        evidence.append("All required Phase 3 conditions are present.")
    else:
        risks.append("Missing one or more required Phase 3 conditions.")
        required_next_actions.append("Run baseline, alpha_zero, real_d, random_d, and shuffled_d.")

    if checks.get("alpha_zero_matches_baseline"):
        evidence.append("alpha_zero matches baseline within configured tolerance.")
    else:
        risks.append("alpha_zero does not match baseline closely enough.")
        required_next_actions.append(
            "Inspect GeoAttention neutral path and shared training controls."
        )

    if checks.get("real_d_beats_baseline"):
        evidence.append("real_d improves final validation loss over baseline.")
    else:
        risks.append("real_d does not improve final validation loss over baseline.")
        required_next_actions.append("Repeat or redesign Phase 3 before graph memory.")

    if checks.get("real_d_beats_random_d"):
        evidence.append("real_d beats random_d control.")
    else:
        risks.append("real_d does not beat random_d control.")
        required_next_actions.append(
            "Verify that relational structure matters beyond arbitrary bias."
        )

    if checks.get("real_d_beats_shuffled_d"):
        evidence.append("real_d beats shuffled_d control.")
    else:
        risks.append("real_d does not beat shuffled_d control.")
        required_next_actions.append(
            "Verify that distance arrangement matters, not only global statistics."
        )

    if checks.get("real_d_validation_loss_finite"):
        evidence.append("real_d validation loss is finite.")
    else:
        risks.append("real_d validation loss is missing or non-finite.")
        required_next_actions.append("Fix numerical stability before deciding the gate.")

    overhead_risk = runtime_overhead_risk(conditions)
    if overhead_risk:
        risks.append(overhead_risk)
        required_next_actions.append("Record and justify runtime overhead before Phase 4.")
    elif runtime_overhead_available(conditions):
        evidence.append("Runtime overhead is recorded and within the default warning threshold.")

    if confirm_seed is not None:
        add_confirm_seed_evidence(confirm_seed, evidence, risks, required_next_actions)

    status = classify_gate(checks, risks, deltas, confirm_seed=confirm_seed)
    if status == "pass":
        next_action = "Proceed to Phase 4: Dynamic Relational Graph Memory."
    elif status == "conditional_pass":
        next_action = "Repeat or strengthen Phase 3 with targeted fixes before Phase 4."
    else:
        next_action = "Do not proceed to Phase 4. Redesign or repeat Phase 3."

    if status != "pass" and not required_next_actions:
        required_next_actions.append(next_action)

    return {
        "gate": "phase3_to_phase4",
        "decision": status,
        "next_action": next_action,
        "evidence": evidence,
        "risks": risks,
        "required_next_actions": dedupe(required_next_actions),
        "inputs": {
            "comparison_conditions": sorted(conditions.keys()),
            "ablation_recommendation": ablation.get("summary", {}).get("recommendation"),
            "confirm_seed_recommendation": (
                confirm_seed.get("summary", {}).get("recommendation") if confirm_seed else None
            ),
        },
        "metrics": {
            "deltas": deltas,
            "ranking": comparison.get("ranking", {}),
        },
        "anti_overclaim": (
            "Passing Gate 1 supports only that induced relational distance can be a "
            "useful attention bias. It does not prove memory, reasoning, or intelligence."
        ),
    }


def add_confirm_seed_evidence(
    confirm_seed: dict[str, Any],
    evidence: list[str],
    risks: list[str],
    required_next_actions: list[str],
) -> None:
    checks = confirm_seed.get("checks", {})
    recommendation = confirm_seed.get("summary", {}).get("recommendation")

    if checks.get("all_losses_finite") and checks.get("all_seeds_equal"):
        evidence.append("Confirm-seed run is finite and uses one shared seed.")
    else:
        risks.append("Confirm-seed run is missing, non-finite, or seed-inconsistent.")
        required_next_actions.append("Repeat confirm-seed validation with consistent seeds.")

    if checks.get("real_d_beats_baseline"):
        evidence.append("Confirm-seed real_d beats baseline.")
    else:
        risks.append("Confirm-seed real_d does not beat baseline.")
        required_next_actions.append("Do not proceed to Phase 4 before repeating Phase 3.")

    if checks.get("real_d_beats_best_random_control"):
        evidence.append("Confirm-seed real_d beats the best random control.")
    else:
        risks.append("Confirm-seed real_d does not beat the best random control.")
        required_next_actions.append(
            "Redesign or repeat Phase 3 before claiming relational geometry advantage."
        )

    if recommendation == "real_d_not_replicated":
        risks.append("Confirm-seed result did not replicate the real_d advantage.")
        required_next_actions.append("Treat Gate 1 as failed until replication improves.")


def classify_gate(
    checks: dict[str, Any],
    risks: list[str],
    deltas: dict[str, Any],
    *,
    confirm_seed: dict[str, Any] | None = None,
) -> GateStatus:
    if confirm_seed is not None and confirm_seed_failed(confirm_seed):
        return "fail"

    required_positive = [
        checks.get("all_required_conditions_present"),
        checks.get("alpha_zero_matches_baseline"),
        checks.get("real_d_beats_baseline"),
        checks.get("real_d_beats_random_d"),
        checks.get("real_d_beats_shuffled_d"),
        checks.get("real_d_validation_loss_finite"),
    ]
    if all(required_positive) and not severe_risks(risks):
        return "pass"

    if not checks.get("all_required_conditions_present") or not checks.get(
        "real_d_validation_loss_finite"
    ):
        return "fail"

    if not checks.get("alpha_zero_matches_baseline"):
        return "fail"

    if checks.get("real_d_beats_random_d") or checks.get("real_d_beats_shuffled_d"):
        return "conditional_pass"

    real_vs_baseline = deltas.get("real_d_vs_baseline", {}).get("final_validation_loss", {})
    if real_vs_baseline.get("absolute") is not None and real_vs_baseline["absolute"] < 0:
        return "conditional_pass"

    return "fail"


def confirm_seed_failed(confirm_seed: dict[str, Any]) -> bool:
    checks = confirm_seed.get("checks", {})
    recommendation = confirm_seed.get("summary", {}).get("recommendation")
    if recommendation == "real_d_not_replicated":
        return True
    if not checks.get("all_losses_finite") or not checks.get("all_seeds_equal"):
        return True
    return not (
        checks.get("real_d_beats_baseline")
        and checks.get("real_d_beats_random_d_alpha_0_2")
        and checks.get("real_d_beats_best_random_control")
    )


def runtime_overhead_available(conditions: dict[str, Any]) -> bool:
    baseline = conditions.get("baseline", {})
    real_d = conditions.get("real_d", {})
    return (
        baseline.get("average_tokens_per_second") is not None
        and real_d.get("average_tokens_per_second") is not None
    )


def runtime_overhead_risk(
    conditions: dict[str, Any],
    *,
    warning_relative_slowdown: float = 0.5,
) -> str | None:
    if not runtime_overhead_available(conditions):
        return "Runtime overhead is not fully recorded."
    baseline_tps = float(conditions["baseline"]["average_tokens_per_second"])
    real_tps = float(conditions["real_d"]["average_tokens_per_second"])
    if baseline_tps <= 0 or real_tps <= 0:
        return "Runtime throughput values are invalid."
    slowdown = (baseline_tps - real_tps) / baseline_tps
    if slowdown > warning_relative_slowdown:
        return f"real_d runtime slowdown is high: {slowdown:.2%}."
    return None


def severe_risks(risks: list[str]) -> list[str]:
    severe_markers = [
        "Missing",
        "does not match baseline",
        "does not improve",
        "does not beat random",
        "does not beat shuffled",
        "missing or non-finite",
        "invalid",
    ]
    return [risk for risk in risks if any(marker in risk for marker in severe_markers)]


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


if __name__ == "__main__":
    main()
