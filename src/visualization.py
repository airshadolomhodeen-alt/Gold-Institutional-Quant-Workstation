import matplotlib.pyplot as plt

def plot_probability_curve(horizons, p_up, p_neutral, p_down):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0A0E17')
    ax.set_facecolor('#131B2E')
    
    ax.plot(horizons, p_up, label='P(UP)', color='#38BDF8', marker='o', linewidth=2)
    ax.plot(horizons, p_neutral, label='P(NEUTRAL)', color='#94A3B8', marker='s', linewidth=2)
    ax.plot(horizons, p_down, label='P(DOWN)', color='#F43F5E', marker='^', linewidth=2)
    
    ax.set_title('Multi-Horizon Marginal Probability Distribution', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Horizon (Candles)', color='#E2E8F0')
    ax.set_ylabel('Probability', color='#E2E8F0')
    ax.tick_params(colors='#E2E8F0')
    ax.legend(facecolor='#131B2E', edgecolor='#334155', labelcolor='#E2E8F0')
    ax.grid(color='#334155', linestyle='--', alpha=0.5)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    return fig
