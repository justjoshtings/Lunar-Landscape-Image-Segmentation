import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

CODE_PATH = os.getcwd()
os.chdir('..')
BASE_PATH = os.getcwd()
os.chdir(CODE_PATH)

results_path = os.path.join(BASE_PATH, 'Results/RESULTS.csv')

res = pd.read_csv(results_path)

loss = res[res.metric.isin(['train_loss', 'val_loss'])]
sns.lineplot(data = loss, x = 'epoch', y = 'value', hue = 'metric')
plt.show()

loss = res[res.metric.isin(['train_bce_with_logits_loss', 'val_bce_with_logits_loss'])]
sns.relplot(data = loss, x = 'epoch', y = 'value', hue = 'model_name', col = 'metric', kind = 'line')
plt.show()