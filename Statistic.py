import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_auc_score

data = pd.read_csv("MSE_ENTROPY.csv")

print(data.mse.describe(percentiles=[0.05, 0.10, 0.12, 0.15, 0.17, 0.20, 0.25, 0.55, 0.75, 0.90, 1]))
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.title('MSE Distribution')
sns.histplot(data.mse)
plt.subplot(1,2,2)
plt.title('Box Plot')
sns.boxplot(x=data.mse)
plt.show()

print(data.entropy.describe(percentiles=[0.05, 0.10, 0.12, 0.15, 0.17, 0.20, 0.25, 0.55, 0.75, 0.90, 1]))
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.title('Entropy Distribution')
sns.histplot(data.entropy)
plt.subplot(1,2,2)
plt.title('Box Plot')
sns.boxplot(x=data.entropy)
plt.show()