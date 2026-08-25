import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py

attention_path = Path("/Volumes/2 TB Backup/runs/physhydra/TRACES-ABP_POSTURES-0-45-90_CHANNELS-RGB_H-300_W-300/tested_on_P034/interpretability_outputs.h5")

data = h5py.File(attention_path,mode='r')
data.keys()
attention_maps = data['attention_maps']
attention_map = attention_maps[1]
plt.imshow(attention_map[0].mean(axis=0))
plt.colorbar()
plt.show()
for i, map in enumerate(attention_map):
    channel_map = map.mean(axis=0)
    plt.imshow(channel_map)
    plt.colorbar()
    plt.title(f'Channel {i} Map')
    plt.show()

bframes = data['B']
plt.imshow(bframes[0,0,:,:])
plt.show()

plt.close('all')
first_key = next(iter(attention))
print(first_key)

scores = channel_scores[first_key][0].numpy()
attention_map = attention[first_key][0].numpy()

reds = attention_map[0]
reds = 1-np.mean(reds, axis=0)
plt.imshow(reds, cmap='hot')
plt.colorbar()
plt.show()


greens = attention_map[1]
greens = (greens - np.min(greens)) / (np.max(greens) - np.min(greens))
blues = attention_map[2]
blues = (blues - np.min(blues)) / (np.max(blues) - np.min(blues))