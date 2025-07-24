import json
import matplotlib.pyplot as plt

# Simple script: load a JSON file and plot all numeric metrics as a bar chart

def main():
    json_path = input("Path to evaluation results JSON: ").strip()
    with open(json_path, 'r') as f:
        results = json.load(f)
    # Only plot numeric metrics
    metrics = {k: v for k, v in results.items() if isinstance(v, (int, float))}
    if not metrics:
        print("No numeric metrics found in the file.")
        return
    plt.bar(metrics.keys(), metrics.values())
    plt.title("Evaluation Metrics")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 