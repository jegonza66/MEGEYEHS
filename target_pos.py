import scipy.io
import matplotlib.image as mpimg
import os
import numpy as np
import matplotlib.pyplot as plt
plt.figure()
plt.close('all')
import pandas as pd

from paths import paths
import load

items_pos_path = paths().item_pos_path()

items_pos = scipy.io.loadmat(items_pos_path)
items_pos_data = items_pos['pos']

items_pos_type = items_pos_data.dtype  # dtypes of structures are "unsized objects"

# * SciPy reads in structures as structured NumPy arrays of dtype object
# * The size of the array is the size of the structure array, not the number
#   elements in any particular field. The shape defaults to 2-dimensional.
# * For convenience make a dictionary of the data using the names from dtypes
# * Since the structure has only one element, but is 2-D, index it at [0, 0]

ndata = {n: items_pos_data[n] for n in items_pos_type.names}
# Reconstruct the columns of the data table from just the time series
# Use the number of intervals to test if a field is a column or metadata
columns = items_pos_type.names
# now make a data frame, setting the time stamps as the index
items_pos = pd.DataFrame(np.concatenate([ndata[c] for c in columns], axis=1), columns=columns)

for key in items_pos.keys():
    key_values = [value[0] for value in items_pos[key].values]
    items_pos[key] = key_values

double_list_keys = list(items_pos.keys())
remove_keys = ['folder', 'item', 'cmp', 'trialabsent']
for key in remove_keys:
    double_list_keys.remove(key)

for key in double_list_keys:
    key_values = [value[0] for value in items_pos[key].values]
    items_pos[key] = key_values


# Complete target pos
plt.ioff()
exp_path = paths().experiment_path()
plots_path = paths().plots_path()

subject = load.subject()
bh_data = subject.beh_data()

all_targets = bh_data[bh_data['Tpres'] == 1]
image_names = all_targets['searchimage'].drop_duplicates()
image_names = image_names.str.split('cmp_', expand=True)[1]
image_names = image_names.str.split('.jpg', expand=True)[0]

target_names = all_targets['st5'].drop_duplicates()

correlations = {}
for image_name, target_name in zip(image_names, target_names):
    print(image_name)

    os.makedirs(plots_path + f'Target_search/{image_name}/', exist_ok=True)

    item_pos_image = items_pos[items_pos['folder'] == image_name]

    target_full = mpimg.imread(exp_path + target_name)
    target = target_full[:, :, :3]
    target_alpha = target_full[:, :, -1]

    image = img = mpimg.imread(exp_path + 'cmp_' + image_name + '.jpg')

    plt.figure()
    plt.imshow(image)
    plt.savefig(plots_path + f'Target_search/{image_name}/search_image.jpg')
    plt.close()

    correlations[image_name] = {}

    for idx, row in item_pos_image.iterrows():
        print(row['indice'])
        x_lims = np.arange(row['pos_x'], min(row['pos_x'] + abs(row['width']), image.shape[1]))
        y_lims = np.arange(row['pos_y'], min(row['pos_y'] + abs(row['height']), image.shape[0]))

        crop_image = image[y_lims, :, :][:, x_lims, :]

        r_corr = np.correlate(crop_image[:, :, 0].ravel(), target[:, :, 0].ravel())[0]
        g_corr = np.correlate(crop_image[:, :, 1].ravel(), target[:, :, 1].ravel())[0]
        b_corr = np.correlate(crop_image[:, :, 2].ravel(), target[:, :, 2].ravel())[0]

        correlations[image_name][row['indice']] = np.mean([r_corr, g_corr, b_corr])

        fig, axs = plt.subplots(2)
        plt.title(image_name)
        axs[0].set_title('Target')
        axs[0].imshow(target)
        axs[1].set_title('Croped item')
        axs[1].imshow(crop_image)
        fig.tight_layout()
        plt.savefig(plots_path + f'Target_search/{image_name}/{row["indice"]}.jpg')
        plt.close(fig)

    print(list(correlations[image_name].values()))
    best_match = item_pos_image.iloc[np.argmax(list(correlations[image_name].values()))]
    print(np.argmax(list(correlations[image_name].values())))

    x_lims = np.arange(best_match['pos_x'], min(best_match['pos_x'] + abs(best_match['width']), image.shape[1]))
    y_lims = np.arange(best_match['pos_y'], min(best_match['pos_y'] + abs(best_match['height']), image.shape[0]))
    crop_image = image[y_lims, :, :][:, x_lims, :]

    fig, axs = plt.subplots(2)
    plt.title(image_name)
    axs[0].set_title('Target')
    axs[0].imshow(target)
    axs[1].set_title('Best match')
    axs[1].imshow(crop_image)

    plt.savefig(plots_path + f'Target_search/{image_name}/Best_match.jpg')
    plt.close(fig)

# target_pos = items_pos[items_pos['istarget'] == 1]