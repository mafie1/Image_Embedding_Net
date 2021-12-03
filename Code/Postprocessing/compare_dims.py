import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from utils_post import load_image
from Code.model import UNet_small, UNet_spoco


HEIGHT = 512
OUT_CHANNELS = 16
EPOCHS = 100
image, mask = load_image(HEIGHT, index = 3)

rel_model_path = '~/Documents/BA_Thesis/Image_Embedding_Net/Code/saved_models/small_UNet/run-dim16-height512-epochs2000/epoch-100-dim16-s512.pt'
model_path = os.path.expanduser(rel_model_path)

loaded_model = UNet_small(in_channels=3, out_channels=OUT_CHANNELS)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()

embedding = loaded_model(image).squeeze(0).detach().numpy()
flat_embedding = embedding.reshape((OUT_CHANNELS, -1))
flat_mask = mask.reshape(-1)


"""Create DataFrame of fully dimensional embeddings"""
df = pd.DataFrame()
df["label"] = flat_mask[:]

for i in range(0, OUT_CHANNELS):
    df['dim{}'.format(i + 1)] = flat_embedding[i][:]

    # filter out background
df_bg_free = df.query('label != 0.0')

print('The variance of the instances is', (df_bg_free.std(), '\n'))
print('The variance of the background is', np.abs(df.std()-df_bg_free.std()))

#df['std background'] = np.array(np.abs(df.std()-df_bg_free.std()))
#df['std instances'] = df_bg_free.std()


# Plot selected dimensions
def scatter_plot():
    df['label'] = df['label'] + 1
    fig = px.scatter(df, x='dim1', y='dim2', color='label', symbol='label', size = 'label',
                        title = 'Instance Pixel Embeddings',
                        width = 1600, height = 800)

    fig.update_layout(legend_orientation="h")


plt.bar(np.linspace(0, OUT_CHANNELS-1, OUT_CHANNELS), sorted(np.array(df_bg_free.std()[1:]), reverse=True), label = 'Background Component', alpha = 0.5)
plt.bar(np.linspace(0, OUT_CHANNELS-1, OUT_CHANNELS), sorted(np.array(df.std()[1:]), reverse=True),
        label = 'Instance Component', alpha = 0.5,
        bottom = sorted(np.array(df_bg_free.std()[1:]), reverse=True ))



plt.xticks(list(np.linspace(0, OUT_CHANNELS-1, OUT_CHANNELS)))
plt.ylabel('Variance per Dimension', size = 'large')
plt.xlabel('Dimensions sorted by Variance', size = 'large')
plt.title('Variance (Spread) Distribution of {} E-Dimensions and {} Training Epochs'.format(OUT_CHANNELS, EPOCHS), size = 'large')
plt.legend()


print(np.array(df_bg_free.std())[1:])
plt.show()
#fig = px.bar(df, x='dim1', y='label')
#fig.show()
