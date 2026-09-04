#!/usr/bin/env python3
"""Realistic industrial defect-rate cost re-weighting analysis.

Context:
In MVTec AD, the test set has ~73% defective parts (1,258 defective, 467 good across 15 categories).
In real factory manufacturing, defect rates (prevalence prior p) are typically 0.1% to 2% (0.001 to 0.02).
At an asymmetric cost ratio of 100:1 (cost of escape vs false alarm), MVTec's 73% test set caused an
inverted evaluation:
  1. The trivial baseline "scrap every part unexamined" cost 467, beating our best detector's 559.
  2. Operating percentiles like p50 and p20 appeared artificially superior to p99 because with 73%
     defective parts and 100:1 escape cost, preventing escapes was weighted 270x heavier than avoiding
     false alarms, completely masking that p50 scraps 60% of all good parts.

This script evaluates the true Expected Cost per part and per 10,000 manufactured parts:
    C = p * FNR(t) * Cost_escape + (1 - p) * FPR(t) * Cost_false_alarm
across:
    - Defect prevalence priors p in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.73]
    - Cost ratios [10:1, 100:1, 1000:1] (Cost_false_alarm = 1.0, Cost_escape = ratio)
    - Threshold percentiles from 20 to 100
    - Multiple backbone arms (A_wrn50_224, B1_dinov2_392, D_wrn50_320, E_resnet50_224)

Outputs:
    outputs/exp_realistic_cost.json
"""
import collections
import json
import os
import sys

# Official MVTec AD test set ground truth counts per category: (n_defective, n_good)
# Total: 1,258 defective, 467 good = 1,725 total test images.
# Defect prevalence in MVTec AD test set = 1258 / 1725 = 0.729275... ~ 73%.
MVTEC_TEST_COUNTS = {
    "bottle": (63, 20),
    "cable": (92, 58),
    "capsule": (109, 23),
    "carpet": (89, 28),
    "grid": (57, 21),
    "hazelnut": (70, 40),
    "leather": (92, 32),
    "metal_nut": (93, 22),
    "pill": (141, 26),
    "screw": (119, 41),
    "tile": (84, 33),
    "toothbrush": (30, 12),
    "transistor": (40, 60),
    "wood": (60, 19),
    "zipper": (119, 32),
}

PRIORS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.73]
COST_RATIOS = [10.0, 100.0, 1000.0]
BATCH_SIZE = 10000  # Parts per evaluation batch


def recover_fn_fp(cat_by_ratio, p_str):
    """Recover average False Negatives (FN) and False Positives (FP) from cost curves.

    In exp_percentile_rule and exp_arms_at_optimal_threshold:
        cost(ratio) = ratio * FN + FP
    With ratio=10 and ratio=100:
        cost(100) - cost(10) = 90 * FN  =>  FN = (cost(100) - cost(10)) / 90
        FP = cost(10) - 10 * FN
    """
    c10 = cat_by_ratio["10.0"]["curve"][p_str]
    c100 = cat_by_ratio["100.0"]["curve"][p_str]
    fn = (c100 - c10) / 90.0
    fp = c10 - 10.0 * fn
    # Handle floating point inaccuracies around zero
    fn = max(0.0, fn)
    fp = max(0.0, fp)
    return fn, fp


def compute_rates(categories_data, percentiles):
    """Compute micro-averaged, macro-averaged, and per-category FNR and FPR curves."""
    total_def = sum(c[0] for c in MVTEC_TEST_COUNTS.values())
    total_good = sum(c[1] for c in MVTEC_TEST_COUNTS.values())

    rates = {
        "percentiles": percentiles,
        "micro": {"fnr": {}, "fpr": {}},
        "macro": {"fnr": {}, "fpr": {}},
        "per_category": {cat: {"fnr": {}, "fpr": {}, "fn": {}, "fp": {}} for cat in categories_data},
    }

    for p in percentiles:
        p_str = str(p)
        tot_fn = 0.0
        tot_fp = 0.0
        cat_fnr_list = []
        cat_fpr_list = []

        for cat, cdata in categories_data.items():
            fn, fp = recover_fn_fp(cdata["by_ratio"], p_str)
            n_def, n_good = MVTEC_TEST_COUNTS[cat]

            cat_fnr = fn / n_def
            cat_fpr = fp / n_good

            rates["per_category"][cat]["fn"][p_str] = round(fn, 4)
            rates["per_category"][cat]["fp"][p_str] = round(fp, 4)
            rates["per_category"][cat]["fnr"][p_str] = round(cat_fnr, 6)
            rates["per_category"][cat]["fpr"][p_str] = round(cat_fpr, 6)

            tot_fn += fn
            tot_fp += fp
            cat_fnr_list.append(cat_fnr)
            cat_fpr_list.append(cat_fpr)

        micro_fnr = tot_fn / total_def
        micro_fpr = tot_fp / total_good
        macro_fnr = sum(cat_fnr_list) / len(cat_fnr_list)
        macro_fpr = sum(cat_fpr_list) / len(cat_fpr_list)

        rates["micro"]["fnr"][p_str] = round(micro_fnr, 6)
        rates["micro"]["fpr"][p_str] = round(micro_fpr, 6)
        rates["macro"]["fnr"][p_str] = round(macro_fnr, 6)
        rates["macro"]["fpr"][p_str] = round(macro_fpr, 6)

    return rates


def compute_expected_costs(rates_dict, percentiles, priors=PRIORS, cost_ratios=COST_RATIOS):
    """Compute expected cost per 10,000 parts across priors, cost ratios, and percentiles."""
    results = {}
    for ratio in cost_ratios:
        r_str = str(ratio)
        results[r_str] = {}
        for p in priors:
            p_str = str(p)
            curve = {}
            for pct in percentiles:
                pct_str = str(pct)
                fnr = rates_dict["fnr"][pct_str]
                fpr = rates_dict["fpr"][pct_str]
                # Expected cost per part: C = p * FNR * ratio + (1 - p) * FPR * 1.0
                cost_per_part = p * fnr * ratio + (1.0 - p) * fpr * 1.0
                cost_10k = cost_per_part * BATCH_SIZE
                curve[pct_str] = round(cost_10k, 2)

            best_pct = min(curve, key=curve.get)
            results[r_str][p_str] = {
                "curve": curve,
                "best_percentile": float(best_pct),
                "best_cost": curve[best_pct],
                "cost_at_p50": curve.get("50") or curve.get("50.0"),
                "cost_at_p95": curve.get("95") or curve.get("95.0"),
                "cost_at_p99": curve.get("99") or curve.get("99.0"),
                "cost_at_p100": curve.get("100") or curve.get("100.0"),
                "p99_vs_p50_ratio": round(curve[str(99)] / curve[str(50)], 3) if ("99" in curve and "50" in curve and curve[str(50)] > 0) else None,
            }
    return results


def compute_baselines(priors=PRIORS, cost_ratios=COST_RATIOS):
    """Compute expected costs for trivial non-inspection policies per 10,000 parts:
    1. Scrap all unexamined: FNR=0, FPR=1. Cost = 10,000 * (1 - p) * Cost_false_alarm
    2. Ship all unexamined: FNR=1, FPR=0. Cost = 10,000 * p * Cost_escape
    """
    baselines = {}
    for ratio in cost_ratios:
        r_str = str(ratio)
        baselines[r_str] = {}
        for p in priors:
            p_str = str(p)
            cost_scrap_all = BATCH_SIZE * (1.0 - p) * 1.0
            cost_ship_all = BATCH_SIZE * p * ratio
            baselines[r_str][p_str] = {
                "scrap_all": round(cost_scrap_all, 1),
                "ship_all": round(cost_ship_all, 1),
                "best_trivial_policy": "scrap_all" if cost_scrap_all < cost_ship_all else "ship_all",
                "best_trivial_cost": min(cost_scrap_all, cost_ship_all),
            }
    return baselines


def main():
    os.makedirs("outputs", exist_ok=True)
    f_rule = "outputs/exp_percentile_rule.json"
    f_arms = "outputs/exp_arms_optimal_threshold.json"

    if not os.path.exists(f_rule) or not os.path.exists(f_arms):
        print(f"Error: Required files {f_rule} or {f_arms} missing!", file=sys.stderr)
        sys.exit(1)

    with open(f_rule, "r") as f:
        data_rule = json.load(f)
    with open(f_arms, "r") as f:
        data_arms = json.load(f)

    # 1. Arm A Fine Grid Analysis (from exp_percentile_rule.json, merged with 20, 30, 40)
    rule_percentiles = data_rule["percentiles"]
    # We also have p=20, 30, 40 from data_arms['arms']['A_wrn50_224']
    armA_cats = data_rule["categories"]
    armA_low_cats = data_arms["arms"]["A_wrn50_224"]["categories"]

    # Build merged categories for Arm A
    merged_armA_cats = {}
    all_armA_percentiles = sorted(set(rule_percentiles + [20, 30, 40]))
    for cat in armA_cats:
        merged_armA_cats[cat] = {"by_ratio": {"10.0": {"curve": {}}, "100.0": {"curve": {}}, "1000.0": {"curve": {}}}}
        for r_str in ["10.0", "100.0", "1000.0"]:
            for p in rule_percentiles:
                merged_armA_cats[cat]["by_ratio"][r_str]["curve"][str(p)] = armA_cats[cat]["by_ratio"][r_str]["curve"][str(p)]
            for p in [20, 30, 40]:
                merged_armA_cats[cat]["by_ratio"][r_str]["curve"][str(p)] = armA_low_cats[cat]["by_ratio"][r_str]["curve"][str(p)]

    rates_armA_fine = compute_rates(merged_armA_cats, all_armA_percentiles)
    expected_costs_armA_fine_micro = compute_expected_costs(rates_armA_fine["micro"], all_armA_percentiles)
    expected_costs_armA_fine_macro = compute_expected_costs(rates_armA_fine["macro"], all_armA_percentiles)

    # 2. Multi-Arm Analysis across 4 arms (from exp_arms_optimal_threshold.json)
    arms_percentiles = data_arms["percentiles"]  # [20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
    arms_analysis = {}

    for arm_tag, arm_info in data_arms["arms"].items():
        rates = compute_rates(arm_info["categories"], arms_percentiles)
        costs_micro = compute_expected_costs(rates["micro"], arms_percentiles)
        costs_macro = compute_expected_costs(rates["macro"], arms_percentiles)
        arms_analysis[arm_tag] = {
            "backbone": arm_info.get("backbone"),
            "img": arm_info.get("img"),
            "mean_auroc": arm_info.get("mean_auroc"),
            "rates": {
                "micro": rates["micro"],
                "macro": rates["macro"],
            },
            "costs_per_10k_micro": costs_micro,
            "costs_per_10k_macro": costs_macro,
        }

    # 3. Baselines
    baselines = compute_baselines()

    # 4. Synthesize comprehensive output structure
    output_summary = {
        "title": "Realistic Industrial Defect-Rate Cost Re-weighting Analysis",
        "priors_defect_prevalence": PRIORS,
        "cost_ratios": COST_RATIOS,
        "batch_size_parts": BATCH_SIZE,
        "dataset_ground_truth": {
            "categories": MVTEC_TEST_COUNTS,
            "total_defective": sum(c[0] for c in MVTEC_TEST_COUNTS.values()),
            "total_good": sum(c[1] for c in MVTEC_TEST_COUNTS.values()),
            "total_test_images": sum(c[0] + c[1] for c in MVTEC_TEST_COUNTS.values()),
            "mvtec_test_prevalence": round(sum(c[0] for c in MVTEC_TEST_COUNTS.values()) / sum(c[0] + c[1] for c in MVTEC_TEST_COUNTS.values()), 6),
        },
        "baselines_per_10k_parts": baselines,
        "arm_A_fine_grid": {
            "tag": "A_wrn50_224",
            "percentiles": all_armA_percentiles,
            "rates": rates_armA_fine,
            "costs_per_10k_micro": expected_costs_armA_fine_micro,
            "costs_per_10k_macro": expected_costs_armA_fine_macro,
        },
        "arms_comparison": arms_analysis,
    }

    out_path = "outputs/exp_realistic_cost.json"
    with open(out_path, "w") as f:
        json.dump(output_summary, f, indent=2)
    print(f"Successfully generated {out_path}\n")

    # Print clear, structured summary tables
    print("=" * 88)
    print("REALISTIC INDUSTRIAL DEFECT-RATE RE-WEIGHTING: ARM A (WRN50 @ 224)")
    print(f"Batch size: {BATCH_SIZE:,} parts | Defect Priors: {PRIORS}")
    print("=" * 88)

    for ratio in COST_RATIOS:
        r_str = str(ratio)
        print(f"\n--- COST RATIO {ratio:.0f}:1 (Cost Escape: {ratio:.0f}, Cost False Alarm: 1) ---")
        print(f"{'Percentile':>10} | " + " | ".join(f"{'p=' + str(pr):>8}" for pr in PRIORS))
        print("-" * 88)

        curve_data = expected_costs_armA_fine_micro[r_str]
        for pct in all_armA_percentiles:
            pct_str = str(pct)
            costs = [f"{curve_data[str(pr)]['curve'][pct_str]:8.1f}" for pr in PRIORS]
            print(f"{pct:10.1f} | " + " | ".join(costs))

        print("-" * 88)
        opt_pcts = [f"{curve_data[str(pr)]['best_percentile']:8.1f}" for pr in PRIORS]
        opt_costs = [f"{curve_data[str(pr)]['best_cost']:8.1f}" for pr in PRIORS]
        p99_costs = [f"{curve_data[str(pr)]['cost_at_p99']:8.1f}" for pr in PRIORS]
        p50_costs = [f"{curve_data[str(pr)]['cost_at_p50']:8.1f}" for pr in PRIORS]
        scrap_all = [f"{baselines[r_str][str(pr)]['scrap_all']:8.1f}" for pr in PRIORS]
        ship_all = [f"{baselines[r_str][str(pr)]['ship_all']:8.1f}" for pr in PRIORS]

        print(f"{'OPTIMAL Pct':>10} | " + " | ".join(opt_pcts))
        print(f"{'Opt Cost':>10} | " + " | ".join(opt_costs))
        print(f"{'Cost @ p99':>10} | " + " | ".join(p99_costs))
        print(f"{'Cost @ p50':>10} | " + " | ".join(p50_costs))
        print(f"{'Scrap All':>10} | " + " | ".join(scrap_all))
        print(f"{'Ship All':>10} | " + " | ".join(ship_all))

    print("\n" + "=" * 88)
    print("ACROSS ALL 4 ARMS AT REALISTIC 1% DEFECT RATE (p=0.01) AND 100:1 COST RATIO:")
    print("=" * 88)
    print(f"{'Arm':<16} | {'AUROC':<7} | {'Opt Pct':>7} | {'Opt Cost':>9} | {'Cost @ p99':>10} | {'Cost @ p50':>10} | {'Scrap All':>9}")
    print("-" * 88)
    for arm_tag, arm_res in arms_analysis.items():
        c100_p01 = arm_res["costs_per_10k_micro"]["100.0"]["0.01"]
        print(f"{arm_tag:<16} | {arm_res['mean_auroc']:<7.4f} | {c100_p01['best_percentile']:>7.1f} | "
              f"{c100_p01['best_cost']:>9.1f} | {c100_p01['cost_at_p99']:>10.1f} | "
              f"{c100_p01['cost_at_p50']:>10.1f} | {baselines['100.0']['0.01']['scrap_all']:>9.1f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
