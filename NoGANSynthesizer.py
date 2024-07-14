import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from joblib import Parallel, delayed

# Step 1: Read and Clean Data
def category_to_integer(category):
    return 1 if category == 'Yes' else 0

# Read data
url = "https://raw.githubusercontent.com/VincentGranville/Main/main/Telecom.csv"  # replace this with your CSV file path
data = pd.read_csv(url)

# Clean and transform data
data['Churn'] = data['Churn'].map(category_to_integer)
data['TotalCharges'].replace(' ', np.nan, inplace=True)
data.dropna(subset=['TotalCharges'], inplace=True)
data['TotalCharges'] = data['TotalCharges'].astype(float)
data['TotalChargeResidues'] = data['TotalCharges'] - data['tenure'] * np.sum(data['TotalCharges']) / np.sum(data['tenure'])

# Select features and split data
features = ['tenure', 'MonthlyCharges', 'TotalChargeResidues', 'Churn']
data = data[features]
data_training = data.sample(frac=0.5, random_state=105)
data_validation = data.drop(data_training.index)

data_training.to_csv('telecom_training.csv', index=False)
data_validation.to_csv('telecom_validation.csv', index=False)

# Step 2: Create Synthetic Data
# Create quantile table
bins_per_feature = [50, 40, 40, 4]
quantile_table = []
for i, feature in enumerate(features):
    quantiles = np.linspace(0, 1, bins_per_feature[i] + 1)
    quantile_table.append(np.quantile(data_training[feature], quantiles))

# Assign observations to bins
def assign_bins(data, quantile_table, bins_per_feature):
    bin_indices = []
    for i, feature in enumerate(features):
        bin_index = np.digitize(data[feature].values, quantile_table[i]) - 1
        bin_index = np.clip(bin_index, 0, bins_per_feature[i] - 1)
        bin_indices.append(bin_index)
    return list(zip(*bin_indices))

data_training['bin_indices'] = assign_bins(data_training, quantile_table, bins_per_feature)

# Generate synthetic data
def generate_synthetic_data(data_training, quantile_table, bins_per_feature, n_samples):
    synthetic_data = []
    bin_counts = data_training['bin_indices'].value_counts()
    for bin_index, count in bin_counts.items():
        lower_bounds = [quantile_table[i][bin_index[i]] for i in range(len(bin_index))]
        upper_bounds = [quantile_table[i][bin_index[i] + 1] for i in range(len(bin_index))]
        for _ in range(count):
            new_sample = [np.random.uniform(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))]
            synthetic_data.append(new_sample)
    return pd.DataFrame(synthetic_data, columns=features)

synthetic_data = generate_synthetic_data(data_training, quantile_table, bins_per_feature, len(data_training))
synthetic_data.to_csv('telecom_synthetic.csv', index=False)

# Step 3: Evaluate Synthetic Data
# Compute ECDF
def compute_ecdf_parallel(data, n_nodes=1000, n_jobs=-1):
    def compute_single_ecdf():
        combo = np.random.uniform(0, 1, len(features))
        combo = combo ** (1 / len(features))
        query = " & ".join([f"{features[i]} <= {np.quantile(data[features[i]], combo[i])}" for i in range(len(features))])
        return {str(combo): len(data.query(query)) / len(data)}

    ecdf_values = Parallel(n_jobs=n_jobs)(delayed(compute_single_ecdf)() for _ in range(n_nodes))
    return {k: v for d in ecdf_values for k, v in d.items()}

validation_ecdf = compute_ecdf_parallel(data_validation)
synthetic_ecdf = compute_ecdf_parallel(synthetic_data)

# Kolmogorov-Smirnov Test
def ks_test(validation_ecdf, synthetic_ecdf):
    ks_distances = []
    for key in validation_ecdf:
        if key in synthetic_ecdf:
            ks_distances.append(abs(validation_ecdf[key] - synthetic_ecdf[key]))
    return max(ks_distances) if ks_distances else None

# Debug prints
print(f"Validation ECDF keys: {list(validation_ecdf.keys())[:5]}")  # Print first 5 keys
print(f"Synthetic ECDF keys: {list(synthetic_ecdf.keys())[:5]}")    # Print first 5 keys

ks_distance = ks_test(validation_ecdf, synthetic_ecdf)
if ks_distance is not None:
    print(f"Kolmogorov-Smirnov distance: {ks_distance}")
else:
    print("No matching keys found between validation and synthetic ECDFs.")

# Step 4: Visualize Results
# Scatter plots
def scatter_plot(data1, data2, feature1, feature2):
    plt.scatter(data1[feature1], data1[feature2], alpha=0.5, label='Synthetic')
    plt.scatter(data2[feature1], data2[feature2], alpha=0.5, label='Real')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.show()

scatter_plot(synthetic_data, data_validation, 'tenure', 'MonthlyCharges')
scatter_plot(synthetic_data, data_validation, 'tenure', 'TotalChargeResidues')
scatter_plot(synthetic_data, data_validation, 'MonthlyCharges', 'TotalChargeResidues')

# Histograms
def hist_plot(data1, data2, feature):
    plt.hist(data1[feature], bins=30, alpha=0.5, label='Synthetic')
    plt.hist(data2[feature], bins=30, alpha=0.5, label='Real')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

hist_plot(synthetic_data, data_validation, 'tenure')
hist_plot(synthetic_data, data_validation, 'MonthlyCharges')
hist_plot(synthetic_data, data_validation, 'TotalChargeResidues')

