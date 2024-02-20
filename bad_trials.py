import os
import pandas as pd
import setup
import load
import save
from paths import paths
import matplotlib.pyplot as plt
import numpy as np
import shutil


save_fig = False
exp_info = setup.exp_info()

all_bad_trials = pd.DataFrame()

for subject_code in exp_info.subjects_ids:

    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    print(f'\nSubject {subject.subject_id}')

    fixations = subject.fixations
    fixations_vs = fixations.loc[fixations['screen'] == 'vs']

    # item vs none fixations
    item_fix_percentage = []
    bad_trials = pd.DataFrame(columns=['Subject', 'Trial', 'Item percentage', 'Image'])
    for trial_idx in range(fixations_vs.trial.max()):
        trial = trial_idx + 1
        trial_fixations = fixations_vs.loc[fixations_vs.trial == trial]
        item_fixations = trial_fixations.loc[trial_fixations.item.notna()]
        # none_fixations = trial_fixations.loc[trial_fixations.item.isnull()]
        if len(trial_fixations):
            item_percentaje = len(item_fixations) / len(trial_fixations) * 100
        else:
            item_percentaje = 0

        if item_percentaje < 100:
            image = subject.trial_imgs[trial_idx]
            bad_trials = pd.concat([bad_trials, pd.DataFrame({'Subject': [subject_code], 'Trial': [trial], 'Item percentage': [item_percentaje], 'Image': [image]})])

    all_bad_trials = pd.concat([all_bad_trials, bad_trials])


all_images = pd.DataFrame()
for image in bad_trials.Image:
    image_average = all_bad_trials['Item percentage'].loc[all_bad_trials.Image == image].mean()
    image_std = all_bad_trials['Item percentage'].loc[all_bad_trials.Image == image].std()
    all_images = pd.concat([all_images, pd.DataFrame({'image': [image], 'percent': [image_average], 'std': [image_std]})])

all_images_sorted = all_images.sort_values(by='percent')

save_path = paths().plots_path() + 'Bad_Trials/'
os.makedirs(save_path, exist_ok=True)
all_images_sorted.to_csv(save_path + 'bad_images_sorted.csv')

fig, ax = plt.subplots()
all_images_sorted['percent'].hist(ax=ax)
save.fig(fig=fig, path=save_path, fname='images histogram', save_svg=False)

bad_images = all_images_sorted.loc[all_images_sorted['percent'] < 40]

for subject_code in exp_info.subjects_ids:
    subject = load.ica_subject(subject_code=subject_code, exp_info=exp_info)
    subject_scanpaths_path = paths().plots_path() + 'Preprocessing/' + subject_code + '/Scanpaths/'
    subject_bad_trials_path = save_path + subject_code + '/'
    os.makedirs(subject_bad_trials_path, exist_ok=True)
    for image in bad_images.image:
        bad_trial_idx = np.where(subject.trial_imgs == image)[0][0] + 1

        image_path = subject_scanpaths_path + f'Trial{bad_trial_idx}.png'

        shutil.copy(src=image_path, dst=subject_bad_trials_path)


