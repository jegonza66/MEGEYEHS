import functions_general
import pandas as pd
import os
import mne
from paths import paths

region_colors = {
        'frontal': '#FFFF00',  # Yellow
        'sensory/motor': '#9B59B6',  # Purple
        'temporal': '#45B7D1',  # Blue
        'parietal': '#00FF00',  # Green
        'occipital': '#FF6B6B',  # Red
        'unknown': '#808080'  # Grey
    }

# Define Subjects_dir as Freesurfer output folder
subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir

parcelation_segmentation = 'aparc.a2009s'  # Parcellation segmentation to use

# Surface labels id by region
region_labels_csv = os.path.join(paths().save_path(), 'aparc.a2009s_regions.csv')  # Path to the CSV file containing region mappings
region_labels = {'aparc': {'occipital': ['cuneus', 'lateraloccipital', 'lingual', 'pericalcarine'],
                       'parietal': ['postcentral', 'superiorparietal', 'supramarginal', 'inferiorparietal', 'precuneus'],
                       'temporal': ['inferiortemporal', 'middletemporal', 'superiortemporal', 'bankssts', 'transversetemporal', 'fusiform', 'entorhinal', 'parahippocampal', 'temporalpole'],
                       'frontal': ['precentral', 'caudalmiddlefrontal', 'superiorfrontal', 'rostralmiddlefrontal', 'lateralorbitofrontal', 'parstriangularis', 'parsorbitalis', 'parsopercularis', 'medialorbitofrontal', 'paracentral', 'frontalpole'],
                       'insula': ['insula'],
                       'cingulate': ['isthmuscingulate', 'posteriorcingulate', 'caudalanteriorcingulate', 'rostralanteriorcingulate']
                           },
                 'aparc.a2009s': functions_general.read_region_labels_csv(csv_path=region_labels_csv, parcellation='aparc.a2009s')
                 }

fsaverage_labels = mne.read_labels_from_annot(subject='fsaverage', parc=parcelation_segmentation, subjects_dir=subjects_dir)
# Remove 'unknown' label for fsaverage aparc labels
if 'aparc' in parcelation_segmentation:
    print("Dropping extra 'unkown' label from lh.")
    drop_idxs = [i for i, label in enumerate(fsaverage_labels) if ('unknown' in label.name.lower() or 'pericallosal' in label.name.lower())]
    for drop_idx in drop_idxs[::-1]:
        fsaverage_labels.pop(drop_idx)

# plot brain regions
hemi = "rh"
for key in region_labels[parcelation_segmentation].keys():
    Brain = mne.viz.get_brain_class()
    brain = Brain("fsaverage", hemi=hemi, surf="pial", views=['lat', 'med'], subjects_dir=subjects_dir, size=(1080, 720))
    color = region_colors[key]
    for region in region_labels['aparc.a2009s'][key]:
        matching_labels = [label for label in fsaverage_labels if region.lower() in label.name.lower() and hemi in label.name.lower()]
        for label in matching_labels:
            brain.add_label(label, borders=False, color=color)

    brain.save_image(filename= paths().plots_path() + f'brain_{key}.png'.replace('/', '_'))