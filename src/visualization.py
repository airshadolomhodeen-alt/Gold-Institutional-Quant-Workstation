# -*- coding: utf-8 -*-
"""
Probability Curve Visualization
"""
import matplotlib.pyplot as plt

def plot_probability_curve(horizons, prob_up, prob_neutral, prob_down):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0A0E17')
    ax.set_facecolor('#131B2E')
    
    ax.plot([f"T+{h}" for h in horizons], prob_up, label="P(UP)", color="#00FF66", marker='o')
    ax.plot([f"T+{h}" for h in horizons], prob_neutral, label="P(NEUTRAL)", color="#94A3B8", marker='s')
    ax.plot([f"T+{h}" for h in horizons], prob_down, label="P(DOWN)", color="#FF3366", marker='^')
    
    ax.set_title("Multi-Horizon Marginal Probability Distribution", color="#F8FAFC", fontsize=12)
    ax.tick_params(colors='#94A3B8')
    ax.legend(facecolor='#131B2E', labelcolor='#F8FAFC')
    ax.grid(color='#1E293B', linestyle='--', alpha=0.5)
    return fig

def plot_predictive_return_distribution():
    fig, ax = plt.subplots(figsize=(10, 3))
    return fig
