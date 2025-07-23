import json

import matplotlib.pyplot as plt

# Collect metrics from all 8 files
metrics = []
filecnt = 5
for i in range(filecnt):
    with open(f"metric_{i}.json") as f:
        data = json.load(f)
        metrics.append(data)

# Extract GPT scores
categories = ["realism", "functionality", "layout", "completion"]
scores = {cat: [] for cat in categories}

for m in metrics:
    for cat in categories:
        try:
            scores[cat].append(m["GPT score (0-10, higher is better)"][cat]["grade"])
        except:
            scores[cat].append(m[cat]["grade"])

# Plotting
plt.figure(figsize=(10, 6))
for cat in categories:
    plt.plot(range(filecnt), scores[cat], label=cat, marker="o")

plt.title("GPT Scores Across Metric Files")
plt.xlabel("Metric File Index")
plt.ylabel("Score (0-10)")
plt.xticks(range(filecnt))
plt.ylim(0, 10.5)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
