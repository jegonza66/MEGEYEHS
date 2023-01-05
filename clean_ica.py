import os
from mne.preprocessing import ICA

from paths import paths
import setup
import load

ica_path = paths().ica_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

display = False

for subject_code in exp_info.subjects_ids:

    # Load data
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    meg_data = subject.load_preproc_meg(preload=True)

    # Downsample
    meg_downsampled = meg_data.copy().pick_types(meg=True)
    meg_downsampled.resample(200)
    meg_downsampled.filter(1, 40)

    # Define ICA
    ica = ICA(method='fastica', random_state=97, n_components=16)

    # Apply ICA
    ica.fit(meg_downsampled)

    if display:
        # Plot sources and components
        ica.plot_sources(meg_downsampled, title='ICA')
        ica.plot_components()

    # Select bad components
    answer = None
    while answer != 'y':
        answer = input('Enter the component numbers to exclude separated by dashes\n'
                       'For example, to exclude 0th 1st and 5th components enter: 0-1-5')

        components = answer.split('-')

        try:
            components = [int(comp) for comp in components]
            answer = input(f'The components to exclude are: {components}\n'
                           f'Is that correct? (y/n)')
        except:
            print(f'Error to convert components to integer values.\n'
                  f'components: {components}\n'
                  f'Please re-enter the components to exclude')
            answer = None

    # Exclude bad components from data
    ica.exclude = components
    meg_ica = meg_data.copy()
    ica.apply(meg_ica)

    # Save ICA clean data
    save_path_ica = ica_path + subject.subject_id + '/'
    os.makedirs(save_path_ica, exist_ok=True)
    path_file_results = os.path.join(save_path_ica, f'Subject_{subject.subject_id}_ICA.fif')
    meg_data.save(path_file_results, overwrite=True)

    if display:
        # Plot to check
        chs = ['MLF14', 'MLF13', 'MLF12', 'MLF11', 'MRF11', 'MRF12', 'MRF13', 'MRF14', 'MZF01']
        chan_idxs = [meg_data.ch_names.index(ch) for ch in chs]

        meg_data.plot(order=chan_idxs, duration=5)
        meg_ica.plot(order=chan_idxs, duration=5)

