#!/usr/bin/env python3
"""
Model Comparison Dashboard
Compares Decision Tree, Random Forest, XGBoost, and Isolation Forest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

plt.style.use('seaborn-v0_8-darkgrid')

print("="*60)
print("MODEL COMPARISON DASHBOARD")
print("="*60)

# Load all model metrics
models_data = {}
model_files = {
    'Decision Tree': 'decision_tree_metrics.json',
    'Random Forest': 'random_forest_metrics.json',
    'XGBoost': 'xgboost_metrics.json'
}

print("\nLoading model metrics...")
for model_name, file_name in model_files.items():
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            models_data[model_name] = json.load(f)
        print(f"  Loaded: {model_name}")
    else:
        print(f"  Warning: {file_name} not found")

# Load Isolation Forest results (anomaly detection)
if os.path.exists('isolation_forest_results.json'):
    with open('isolation_forest_results.json', 'r') as f:
        isolation_data = json.load(f)
    print(f"  Loaded: Isolation Forest")
else:
    isolation_data = None
    print(f"  Warning: isolation_forest_results.json not found")

# Create comparison dataframe
print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)

targets = ['pm25', 'pm10', 'co', 'lpg', 'hydrogen']
metrics = ['test_mae', 'test_rmse', 'test_r2']

comparison_data = []
for model_name, data in models_data.items():
    for target in targets:
        if target in data:
            comparison_data.append({
                'Model': model_name,
                'Target': target.upper(),
                'MAE': data[target]['test_mae'],
                'RMSE': data[target]['test_rmse'],
                'R2': data[target]['test_r2']
            })

comparison_df = pd.DataFrame(comparison_data)

# Display comparison table
print("\nComparison Table (Test Set Performance):")
print("-" * 80)
print(f"{'Model':<15} {'Target':<10} {'MAE':<12} {'RMSE':<12} {'R2':<12}")
print("-" * 80)

for model_name in models_data.keys():
    model_df = comparison_df[comparison_df['Model'] == model_name]
    for _, row in model_df.iterrows():
        print(f"{row['Model']:<15} {row['Target']:<10} {row['MAE']:<12.4f} {row['RMSE']:<12.4f} {row['R2']:<12.4f}")
    if model_name != list(models_data.keys())[-1]:
        print("-" * 80)

print("-" * 80)

# Find best model for each target by R2
print("\n" + "="*60)
print("BEST MODEL FOR EACH TARGET (by R2 Score)")
print("="*60)

for target in targets:
    target_upper = target.upper()
    target_data = comparison_df[comparison_df['Target'] == target_upper]
    if len(target_data) > 0:
        best_model = target_data.loc[target_data['R2'].idxmax()]
        print(f"\n{target_upper}:")
        print(f"   Best Model: {best_model['Model']}")
        print(f"   R2 Score: {best_model['R2']:.4f}")
        print(f"   MAE: {best_model['MAE']:.4f}")
        print(f"   RMSE: {best_model['RMSE']:.4f}")

# Generate comparison visualizations
print("\n" + "="*60)
print("Generating Comparison Visualizations")
print("="*60)

# 1. R2 Comparison Bar Chart
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(targets))
width = 0.25
colors = ['#667eea', '#48bb78', '#ed8936']

for i, (model_name, color) in enumerate(zip(models_data.keys(), colors)):
    r2_values = []
    for target in targets:
        if target in models_data[model_name]:
            r2_values.append(models_data[model_name][target]['test_r2'])
        else:
            r2_values.append(0)

    offset = (i - 1) * width
    bars = ax.bar(x + offset, r2_values, width, label=model_name, color=color, alpha=0.8)

    # Add value labels
    for bar, val in zip(bars, r2_values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Target Variable', fontsize=12)
ax.set_ylabel('R2 Score', fontsize=12)
ax.set_title('Model Comparison: R2 Scores by Target', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([t.upper() for t in targets])
ax.legend(loc='lower right')
ax.set_ylim(0, 1)
ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (0.8)')
ax.axhline(y=0.6, color='yellow', linestyle='--', alpha=0.5, label='Good (0.6)')
ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Fair (0.4)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison_r2.png', dpi=150, bbox_inches='tight')
print("Saved: model_comparison_r2.png")
plt.close()

# 2. MAE Comparison Bar Chart
fig, ax = plt.subplots(figsize=(14, 8))

for i, (model_name, color) in enumerate(zip(models_data.keys(), colors)):
    mae_values = []
    for target in targets:
        if target in models_data[model_name]:
            mae_values.append(models_data[model_name][target]['test_mae'])
        else:
            mae_values.append(0)

    offset = (i - 1) * width
    bars = ax.bar(x + offset, mae_values, width, label=model_name, color=color, alpha=0.8)

    # Add value labels
    for bar, val in zip(bars, mae_values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)

ax.set_xlabel('Target Variable', fontsize=12)
ax.set_ylabel('MAE (Mean Absolute Error)', fontsize=12)
ax.set_title('Model Comparison: MAE by Target (Lower is Better)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([t.upper() for t in targets])
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison_mae.png', dpi=150, bbox_inches='tight')
print("Saved: model_comparison_mae.png")
plt.close()

# 3. RMSE Comparison Bar Chart
fig, ax = plt.subplots(figsize=(14, 8))

for i, (model_name, color) in enumerate(zip(models_data.keys(), colors)):
    rmse_values = []
    for target in targets:
        if target in models_data[model_name]:
            rmse_values.append(models_data[model_name][target]['test_rmse'])
        else:
            rmse_values.append(0)

    offset = (i - 1) * width
    bars = ax.bar(x + offset, rmse_values, width, label=model_name, color=color, alpha=0.8)

    # Add value labels
    for bar, val in zip(bars, rmse_values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)

ax.set_xlabel('Target Variable', fontsize=12)
ax.set_ylabel('RMSE (Root Mean Square Error)', fontsize=12)
ax.set_title('Model Comparison: RMSE by Target (Lower is Better)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([t.upper() for t in targets])
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison_rmse.png', dpi=150, bbox_inches='tight')
print("Saved: model_comparison_rmse.png")
plt.close()

# 4. Radar Chart for Model Comparison
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Average R2 across all targets for each model
model_avg_r2 = {}
for model_name, data in models_data.items():
    r2_values = []
    for target in targets:
        if target in data:
            r2_values.append(data[target]['test_r2'])
    if r2_values:
        model_avg_r2[model_name] = np.mean(r2_values)

# Radar chart
categories = list(model_avg_r2.keys())
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # Close the loop

values = list(model_avg_r2.values())
values += values[:1]  # Close the loop

ax.plot(angles, values, 'o-', linewidth=2, color='#667eea')
ax.fill(angles, values, alpha=0.25, color='#667eea')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
ax.set_title('Model Comparison: Average R2 Score Across All Targets',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True)

plt.tight_layout()
plt.savefig('model_comparison_radar.png', dpi=150, bbox_inches='tight')
print("Saved: model_comparison_radar.png")
plt.close()

# 5. Performance Heatmap
fig, ax = plt.subplots(figsize=(12, 8))

# Create heatmap data
heatmap_data = []
row_labels = []
for model_name in models_data.keys():
    for target in targets:
        if target in models_data[model_name]:
            heatmap_data.append([
                models_data[model_name][target]['test_r2'],
                models_data[model_name][target]['test_mae'],
                models_data[model_name][target]['test_rmse']
            ])
            row_labels.append(f"{model_name}\n{target.upper()}")
        else:
            heatmap_data.append([0, 0, 0])
            row_labels.append(f"{model_name}\n{target.upper()}")

heatmap_array = np.array(heatmap_data)
im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['R2 (↑)', 'MAE (↓)', 'RMSE (↓)'], fontsize=11)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=9)
ax.set_title('Model Performance Heatmap\n(Green = Better, Red = Worse)',
             fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Normalized Score', fontsize=10)

# Add value annotations
for i in range(len(row_labels)):
    for j in range(3):
        text = ax.text(j, i, f'{heatmap_array[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()
plt.savefig('model_comparison_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: model_comparison_heatmap.png")
plt.close()

# 6. Isolation Forest Summary (if available)
if isolation_data:
    print("\n" + "="*60)
    print("ISOLATION FOREST - ANOMALY DETECTION SUMMARY")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Anomaly ratio by pollutant
    pollutants = list(isolation_data['anomaly_statistics'].keys())
    ratios = [isolation_data['anomaly_statistics'][p]['ratio'] for p in pollutants]

    colors_bar = ['red' if r > 1 else 'green' for r in ratios]
    bars = axes[0].bar(pollutants, ratios, color=colors_bar, alpha=0.7)
    axes[0].axhline(y=1, color='black', linestyle='--', linewidth=2)
    axes[0].set_ylabel('Anomaly Ratio (Anomaly Mean / Normal Mean)')
    axes[0].set_title('Anomaly Impact by Pollutant')
    axes[0].set_xticklabels([p.upper() for p in pollutants], rotation=45)
    axes[0].grid(True, alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, ratios):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}x', ha='center', va='bottom', fontsize=10)

    # Pie chart of anomaly distribution
    sizes = [isolation_data['normal_samples'], isolation_data['anomaly_samples']]
    labels = [f'Normal\n{sizes[0]:,} (95%)', f'Anomaly\n{sizes[1]:,} (5%)']
    colors_pie = ['#48bb78', '#e53e3e']
    axes[1].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                startangle=90, explode=(0, 0.1))
    axes[1].set_title('Anomaly Distribution')

    plt.suptitle('Isolation Forest: Anomaly Detection Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('isolation_forest_summary.png', dpi=150, bbox_inches='tight')
    print("Saved: isolation_forest_summary.png")
    plt.close()

# Generate HTML report
print("\nGenerating HTML comparison report...")

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Air Quality Model Comparison</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .best {{
            background-color: #d4edda;
            font-weight: bold;
        }}
        .metric-card {{
            display: inline-block;
            width: 200px;
            margin: 10px;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
        }}
        img {{
            width: 100%;
            max-width: 800px;
            margin: 20px auto;
            display: block;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌍 Air Quality Prediction - Model Comparison Dashboard</h1>
        <p style="text-align: center">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>📊 Model Performance Comparison</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Target</th>
                <th>MAE ↓</th>
                <th>RMSE ↓</th>
                <th>R² ↑</th>
            </tr>
"""

# Add table rows
for model_name in models_data.keys():
    model_df = comparison_df[comparison_df['Model'] == model_name]
    first_row = True
    for _, row in model_df.iterrows():
        r2_class = 'best' if row['R2'] == max(comparison_df[comparison_df['Target'] == row['Target']]['R2']) else ''
        html_content += f"""
            <tr>
                <td>{model_name if first_row else ''}</td>
                <td>{row['Target']}</td>
                <td>{row['MAE']:.4f}</td>
                <td>{row['RMSE']:.4f}</td>
                <td class="{r2_class}">{row['R2']:.4f}</td>
            </tr>"""
        first_row = False

html_content += """
        </table>

        <h2>📈 Visualization Gallery</h2>
        <h3>R2 Score Comparison</h3>
        <img src="model_comparison_r2.png" alt="R2 Comparison">

        <h3>MAE Comparison</h3>
        <img src="model_comparison_mae.png" alt="MAE Comparison">

        <h3>RMSE Comparison</h3>
        <img src="model_comparison_rmse.png" alt="RMSE Comparison">

        <h3>Radar Chart (Average R2)</h3>
        <img src="model_comparison_radar.png" alt="Radar Chart">

        <h3>Performance Heatmap</h3>
        <img src="model_comparison_heatmap.png" alt="Heatmap">
"""

if isolation_data:
    html_content += """
        <h2>🔍 Isolation Forest - Anomaly Detection</h2>
        <img src="isolation_forest_summary.png" alt="Isolation Forest Summary">

        <div style="text-align: center; margin: 20px;">
            <div class="metric-card">
                <div class="metric-value">""" + f"{isolation_data['anomaly_percentage']:.1f}%" + """</div>
                <div class="metric-label">Anomaly Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">""" + f"{isolation_data['anomaly_samples']:,}" + """</div>
                <div class="metric-label">Total Anomalies</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">""" + f"{isolation_data['anomaly_statistics']['pm10']['ratio']:.2f}x" + """</div>
                <div class="metric-label">PM10 Anomaly Factor</div>
            </div>
        </div>
"""

html_content += f"""
        <h2>🏆 Best Models Summary</h2>
        <table>
            <tr>
                <th>Target</th>
                <th>Best Model</th>
                <th>R² Score</th>
            </tr>
"""

for target in targets:
    target_upper = target.upper()
    target_data = comparison_df[comparison_df['Target'] == target_upper]
    if len(target_data) > 0:
        best = target_data.loc[target_data['R2'].idxmax()]
        html_content += f"""
            <tr>
                <td>{target_upper}</td>
                <td>{best['Model']}</td>
                <td>{best['R2']:.4f}</td>
            </tr>
        """

html_content += f"""
        </table>

        <div class="footer">
            <p>Models trained on {comparison_df.shape[0] // len(models_data)} records per model</p>
            <p>Decision Tree • Random Forest • XGBoost • Isolation Forest (Anomaly Detection)</p>
        </div>
    </div>
</body>
</html>
"""

with open('model_comparison_dashboard.html', 'w') as f:
    f.write(html_content)
print("Saved: model_comparison_dashboard.html")

# Final summary
print("\n" + "="*60)
print("MODEL COMPARISON DASHBOARD COMPLETE!")
print("="*60)

print("\nFiles Generated:")
print("   Visualizations:")
print("   - model_comparison_r2.png")
print("   - model_comparison_mae.png")
print("   - model_comparison_rmse.png")
print("   - model_comparison_radar.png")
print("   - model_comparison_heatmap.png")
if isolation_data:
    print("   - isolation_forest_summary.png")
print("   - model_comparison_dashboard.html (Open this in browser)")

print("\n" + "="*60)
print("Open 'model_comparison_dashboard.html' in your browser")
print("to view the complete comparison dashboard!")
print("="*60)
