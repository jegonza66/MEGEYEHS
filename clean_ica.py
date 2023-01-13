import os
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import mne
import pathlib
import save
from paths import paths
import setup
import load

fig = plt.figure()
plt.close(fig)

preproc_path = paths().preproc_path()
ica_path = paths().ica_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

display = True

for subject_code in exp_info.subjects_ids[4:6]:print(subject.subject_id)

    # Load data
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    meg_data = subject.load_preproc_meg()

    # Downsample and filter
    sfreq = 200
    lfreq = 1
    hfreq = 40
    downsampled_path = pathlib.Path(os.path.join(preproc_path, subject.subject_id, f'down-filt_meg({sfreq}_{lfreq}_{hfreq}).fif'))
    loaded_data = False

    try:
        # Load
        meg_downsampled = mne.io.read_raw_fif(downsampled_path, preload=False)
        loaded_data = True
        print('Downsampled data laoded')
    except:
        # Compute and save
        print('Downsampling and filtering...')
        meg_downsampled = meg_data.copy().pick_types(meg=True)
        meg_downsampled.resample(sfreq)
        meg_downsampled.filter(lfreq, hfreq)
        meg_downsampled.save(downsampled_path, overwrite=True)

    # ICA
    save_path_ica = ica_path + subject.subject_id + '/'
    ica_fname = 'ICA.pkl'

    try:
        if loaded_data:
            # Load ICA
            ica = load.var(file_path=save_path_ica + ica_fname)
            print('ICA object loaded')
        else:
            raise ValueError('No loaded data. Running ICA on new data')
    except:
        # Define ICA
        print('Running ICA...')
        ica_components = 64
        ica = ICA(method='fastica', random_state=97, n_components=ica_components)

        # Apply ICA
        ica.fit(meg_downsampled)

        # Save ICA
        os.makedirs(save_path_ica, exist_ok=True)
        save.var(var=ica, path=save_path_ica, fname=ica_fname)

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

    # Save components figures
    if display:
        # Create directory
        fig_path = plot_path + f'ICA/{subject.subject_id}/'
        os.makedirs(fig_path, exist_ok=True)

        # Plot properties of excluded components
        ica.plot_properties(meg_downsampled, picks=components, show=False)

        # Get figures
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for i, fig in enumerate(figs):
            save.fig(fig=fig, path=fig_path, fname=f'figure_{i}')

    # Exclude bad components from data
    ica.exclude = components
    subject.ex_components = components
    meg_ica = meg_data.copy()
    meg_ica.load_data()
    ica.apply(meg_ica)

    # Save ICA clean data
    save_path_ica = ica_path + subject.subject_id + '/'
    os.makedirs(save_path_ica, exist_ok=True)
    path_file = os.path.join(save_path_ica, f'Subject_{subject.subject_id}_ICA.fif')
    meg_ica.save(path_file, overwrite=True)

    # Save subject
    save.var(var=subject, path=save_path_ica, fname='Subject_data.pkl')

    if display:
        # Plot to check
        chs = ['MLF14', 'MLF13', 'MLF12', 'MLF11', 'MRF11', 'MRF12', 'MRF13', 'MRF14', 'MZF01']
        chan_idxs = [meg_data.ch_names.index(ch) for ch in chs]

        meg_data.plot(order=chan_idxs, duration=5)
        meg_ica.plot(order=chan_idxs, duration=5)

