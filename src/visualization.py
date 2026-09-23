# src/visualization.py
import matplotlib.pyplot as plt

def plot_probability_curve(horizons, prob_up, prob_neutral, prob_down):
    """Generates the multi-horizon probability distribution chart."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(horizons, prob_up, marker='o', color='#00FF66', label='P(UP)')
    ax.plot(horizons, prob_neutral, marker='s', color='#FF9900', label='P(NEUTRAL)')
    ax.plot(horizons, prob_down, marker='^', color='#FF3366', label='P(DOWN)')
    ax.set_title("Multi-Horizon Probabilistic Distribution (t+1 to t+10)", color='#F8FAFC', fontweight='bold')
    ax.set_xlabel("Forecast Horizon (Candles)", color='#E2E8F0')
    ax.set_ylabel("Probability", color='#E2E8F0')
    ax.set_facecolor('#131B2E')
    fig.patch.set_facecolor('#0A0E17')
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(facecolor='#131B2E', edgecolor='#1E293B')
    return fig
