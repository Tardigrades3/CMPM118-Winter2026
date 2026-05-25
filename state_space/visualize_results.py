import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def visualize_results(json_path):
    # 1. Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 2. Extract Immediate and Final performance data
    imm_data = data['immediate_performance']
    fin_data = data['final_performance']
    
    # 3. Flatten into a DataFrame for Seaborn
    rows = []
    for subject in sorted(imm_data.keys(), key=lambda x: int(x.split('_')[1])):
        rows.append({
            "Subject": subject,
            "Accuracy": imm_data[subject]['accuracy'],
            "Type": "Immediate"
        })
        rows.append({
            "Subject": subject,
            "Accuracy": fin_data[subject]['accuracy'],
            "Type": "Final (After CL)"
        })
    
    df = pd.DataFrame(rows)
    
    # 4. Create the Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 6))
    
    # Use a bar plot to show the delta between immediate and final
    ax = sns.barplot(data=df, x="Subject", y="Accuracy", hue="Type", palette="viridis")
    
    # Customizing the visual
    plt.title(f"Catastrophic Forgetting Analysis: {data['metadata']['mode']} mode", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.ylabel("Classification Accuracy")
    plt.tight_layout()
    
    # 5. Show and Save
    output_filename = json_path.replace('.json', '.png')
    plt.savefig(output_filename, dpi=300)
    print(f"Visualization saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', help="Path to the evaluation JSON file")
    args = parser.parse_args()
    
    visualize_results(args.json_path)