import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load data ---
data = pd.read_csv("ex3_largecase.csv")
data['Timestamp'] = pd.to_datetime(data['Timestamp'], utc=True)

# Basic info
print(data.head())
print(data.describe())

# --- Separate the 6 storm events ---
events = data['Event_ID'].unique()
print(f"Found {len(events)} events: {events}")

# --- Plot each event ---
sns.set_style("whitegrid")
fig, axes = plt.subplots(len(events), 1, figsize=(10, 2.5 * len(events)), sharex=True)

for i, e in enumerate(events):
    df = data[data['Event_ID'] == e]
    t = df['Timestamp']
    ax = axes[i] if len(events) > 1 else axes
    
    ax2 = ax.twinx()
    ax.plot(t, df['Volume'], color='steelblue', label='Storage Volume [m³]')
    ax2.plot(t, df['Rainfall'] * 5, color='orange', label='Rainfall (µm/min ×5min)', alpha=0.6)
    ax2.plot(t, df['Pumpflow'], color='green', label='Pump Flow [m³/min]', alpha=0.6)
    
    ax.set_ylabel("Volume [m³]", color='steelblue')
    ax2.set_ylabel("Rainfall / Pumpflow", color='gray')
    ax.set_title(f"Event {e}")
    
    if i == len(events) - 1:
        ax.set_xlabel("Time (UTC)")
        
    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

plt.tight_layout()
plt.show()
