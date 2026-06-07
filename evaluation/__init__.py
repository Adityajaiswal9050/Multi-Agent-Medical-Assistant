"""
evaluation/
-----------
Systematic threshold evaluation for the RAG agent handoff decision.

Modules:
    evaluate_thresholds.py  — runs evaluation across threshold range 0.30-0.70,
                               produces plots and threshold_report.txt
    agent_router.py         — AgentRouter class with tuned DEFAULT_THRESHOLD=0.60

Run evaluation:
    cd evaluation/
    pip install numpy scikit-learn matplotlib seaborn
    python evaluate_thresholds.py
"""

from evaluation.agent_router import AgentRouter, AgentAction, RoutingDecision

__all__ = ["AgentRouter", "AgentAction", "RoutingDecision"]
