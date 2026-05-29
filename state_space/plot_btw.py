import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bwt_matrix(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    imm_data = data['immediate_performance']
    fin_data = data['final_performance']
    
    tasks = list(imm_data.keys())
    num_tasks = len(tasks)
    
    # Create the Transfer Matrix
    # Matrix[i][j] = Accuracy on Task j after training on Task i
    # Since we only have Immediate (diagonal) and Final (bottom row), 
    # we will visualize the delta (Backward Transfer)
    
    bwt_values = []
    task_labels = []
    
    for task in tasks:
        imm_acc = imm_data[task]['accuracy']
        fin_acc = fin_data[task]['accuracy']
        # Backward Transfer = Final Accuracy - Immediate Accuracy
        # Negative means forgetting. Positive means the new task actually helped the old one!
        bwt = (fin_acc - imm_acc) * 100 
        bwt_values.append(bwt)
        # Clean up labels for the plot
        task_labels.append(task.replace('subject_', 'S').replace('Subject_', 'S').replace('_Exercise_', ' E'))

    # Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Color code: Red for forgetting, Green for positive transfer
    colors = ['#e74c3c' if val < 0 else '#2ecc71' for val in bwt_values]
    
    bars = plt.bar(task_labels, bwt_values, color=colors)
    
    plt.axhline(0, color='black', linewidth=1.5)
    plt.title(f"Backward Transfer (BWT) per Task\nMode: {data['metadata']['mode'].upper()}", fontsize=16, pad=20)
    plt.ylabel("Accuracy Delta (%)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add the numbers on top of the bars
    for bar in bars:
        yval = bar.get_height()
        offset = 2 if yval >= 0 else -3
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset, f"{yval:.1f}%", 
                 ha='center', va='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    output_filename = json_path.replace('.json', '_bwt.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"BWT chart saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', help="Path to the evaluation JSON file")
    args = parser.parse_args()
    plot_bwt_matrix(args.json_path)