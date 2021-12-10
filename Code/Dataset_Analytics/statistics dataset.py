import os
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
from time import time
import torch
import random
from Preprocessing.dataset_plants_multiple import CustomDatasetMultiple, image_train_transform, mask_train_transform
from Postprocessing.utils_post import load_val_image

start_time = time()

torch.manual_seed(0)
random.seed(0)

HEIGHT, WIDTH = 512, 512

dir = os.path.expanduser('~/Documents/BA_Thesis/CVPPP2017_instances/training/A1')
contents = os.listdir(dir)
images = list(filter(lambda k: 'rgb' in k, contents))

print('The Dataset A1 contains ', len(images), ' images')

Plants = CustomDatasetMultiple(dir,
                               transform=None,
                               image_transform=image_train_transform(HEIGHT, WIDTH),
                               mask_transform=mask_train_transform(HEIGHT, WIDTH)
                               )

train_set, val_set, test_set = torch.utils.data.random_split(Plants, [80, 28, 20])

def get_smallest_instance():
    smallest = 0.5

    for index, single_image in enumerate(images):
        mask_path = os.path.join(dir, images[index].replace('rgb.png', 'label.png'))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        uniques, counts = np.unique(mask, return_counts=True)

        if np.min(counts)/counts.sum() <= smallest:
            smallest = np.min(counts)/counts.sum()

    return smallest

print(get_smallest_instance())

def get_instance_count():
    number_labels = []
    for index, single_image in enumerate(images):
        mask_path = os.path.join(dir, images[index].replace('rgb.png', 'label.png'))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        uniques = np.unique(mask)
        number_labels = np.append(number_labels, len(uniques))

    return number_labels

def get_bg_fg_ratio():
    ratios = []

    for index, single_image in enumerate(images):
        mask_path = os.path.join(dir, images[index].replace('rgb.png', 'label.png'))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        uniques, counts = np.unique(mask, return_counts=True)

        bg = counts[0]
        fg = np.sum(counts[1:])

        ratio = fg/bg
        ratios = np.append(ratios, ratio)

    return(ratios)


df = pd.DataFrame(data = get_bg_fg_ratio(), columns=['ratio'])
print('Average Ratio:', df.mean())

df['number of leaves'] = get_instance_count()
print(df.head(5))


fig = px.histogram(df, x = "ratio",
                   nbins = 100,
                   range_x = [0, 1],
                   #title = 'Histogram of the Foreground-Background Ratios',
                   template= 'seaborn',
                   histnorm='percent')


fig_2 = px.histogram(df, x = 'number of leaves',
                     nbins = 20 ,
                     #title = 'Histogram of Number of Leaves in Training Images',
                     histnorm='percent',
                     range_x = [11, 21],
                     template = 'seaborn')
#occurance within dataset
fig_2.update_xaxes(tick0=12.0, dtick=1)

fig.write_image("images_statistics/ratio_histogram.png", scale = 3)
fig_2.write_image('images_statistics/leave_count_histogram.png', scale = 3)

#fig.write_image('sample.png')

end_time = time()

print('The script took ', end_time-start_time, 'sec to run')