"""
Interactive Demonstration and Verification of UCA Controller.

Usage:
    python src/experiments/demo_uca.py
    python src/experiments/demo_uca.py --query "What is the capital of Japan?"
    python src/experiments/demo_uca.py --query "Prove that every continuous function on [a, b] is Riemann integrable."
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.uca_controller import UCAController


def main():
    parser = argparse.ArgumentParser(description="Demonstrate UCA Adaptive Controller")
    parser.add_argument("--query", type=str, default=None, help="Custom query string")
    parser.add_argument("--mock", action="store_true", default=True, help="Use mock mode for fast CPU testing")
    args = parser.parse_args()

    uca = UCAController(mock_mode=args.mock)

    test_queries = [
        ("What is the capital of Japan?", "easy"),
        ("In what year did the Apollo 11 mission land on the Moon?", "medium"),
        ("Explain how transformers scale with sequence length and memory footprint.", "medium"),
        ("Prove that every planar graph is 4-colorable using algebraic topology.", "hard"),
        ("Nonsense impossible fictional paradox query xyzz123", "hard"),
    ]

    if args.query:
        test_queries = [(args.query, None)]

    print("==========================================================================================")
    print("                     UCA ADAPTIVE COMPRESSION CONTROLLER DEMO")
    print("==========================================================================================")
    print(f"{'Query':<48} | {'Route':<10} | {'IC':<5} | {'UCA':<5} | {'KV%':<5} | {'Beam':<4} | {'Tiers Used'}")
    print("-" * 105)

    for q, diff in test_queries:
        res = uca.compute_compression_config(q, difficulty=diff)
        route = res["route"]
        ic = f"{res['ic_score']:.2f}"
        uca_val = f"{res['uca_score']:.2f}"
        kv = f"{int(res['kv_budget'] * 100)}%"
        beam = str(res["beam_width"])
        tiers = res["tier_used"]
        q_short = (q[:45] + "...") if len(q) > 48 else q

        print(f"{q_short:<48} | {route:<10} | {ic:<5} | {uca_val:<5} | {kv:<5} | {beam:<4} | {tiers}")

    print("==========================================================================================")


if __name__ == "__main__":
    main()
