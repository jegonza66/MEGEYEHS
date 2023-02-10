import os
import time
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import mne
import pathlib
import save
from paths import paths
import setup
import load
import functions_analysis
import functions_general
import numpy as np
# plt.figure()
# plt.close('all')

preproc_path = paths().preproc_path()
ica_path = paths().ica_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

display = True

for subject_code in exp_info.subjects_ids:

    # Load data
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    meg_data = subject.load_preproc_meg()

    # Downsample and filter
    sfreq = 200
    lfreq = 1
    hfreq = 40
    downsampled_path = pathlib.Path(os.path.join(preproc_path, subject.subject_id, f'down-filt({sfreq}_{lfreq}_{hfreq})_meg.fif'))
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
        ica.plot_sources(meg_downsampled, title='ICA', theme='dark')
        ica.plot_components()



    # Ploch's algorithm for saccadic artifacts detection by variance comparison

    # Sac ID
    emap_sac_id = 'sac_emap'
    # Fix ID
    emap_fix_id = 'fix_emap'
    # Screen
    screen = functions_general.get_screen(epoch_id=emap_sac_id)
    # MSS
    mss = functions_general.get_mss(epoch_id=emap_sac_id)
    # Item
    tgt = functions_general.get_item(epoch_id=emap_sac_id)
    # Saccades direction
    dir = functions_general.get_dir(epoch_id=emap_sac_id)

    # Define events
    print('Saccades')
    sac_metadata, sac_events, sac_events_id, sac_metadata_sup = \
        functions_analysis.define_events(subject=subject, epoch_id=emap_sac_id, screen=screen, mss=mss, dur=None,
                                         tgt=tgt, dir=dir, meg_data=meg_data)

    print('Fixations')
    fix_metadata, fix_events, fix_events_id, fix_metadata_sup = \
        functions_analysis.define_events(subject=subject, epoch_id=emap_fix_id, screen=screen, mss=mss, dur=None,
                                         tgt=tgt, dir=dir, meg_data=meg_data)

    # Get time windows from epoch_id name
    sac_tmin = -0.005  # Add previous 5 ms
    sac_tmax = sac_metadata_sup['duration'].min() + 0.01  # Min sac duration + 10 ms
    fix_tmin = 0
    fix_tmax = fix_metadata_sup['duration'].min()  # Min sac duration + 10 ms

    # Epoch data
    sac_epochs = mne.Epochs(raw=meg_data, events=sac_events, event_id=sac_events_id, tmin=sac_tmin, tmax=sac_tmax, reject=None,
                        event_repeated='drop', metadata=sac_metadata, preload=True, baseline=(0,0))
    fix_epochs = mne.Epochs(raw=meg_data, events=fix_events, event_id=fix_events_id, tmin=fix_tmin, tmax=fix_tmax, reject=None,
                            event_repeated='drop', metadata=fix_metadata, preload=True, baseline=(0,0))

    # Append saccades df as epochs metadata
    if sac_metadata_sup is not None:
        sac_metadata_sup = sac_metadata_sup.loc[(sac_metadata_sup['id'].isin(sac_epochs.metadata['event_name']))].reset_index(drop=True)
        sac_epochs.metadata = sac_metadata_sup
    if fix_metadata_sup is not None:
        fix_metadata_sup = fix_metadata_sup.loc[(fix_metadata_sup['id'].isin(fix_epochs.metadata['event_name']))].reset_index(drop=True)
        fix_epochs.metadata = fix_metadata_sup

    # Get the ICA sources for the epoched data
    sac_ica_sources = ica.get_sources(sac_epochs)
    fix_ica_sources = ica.get_sources(fix_epochs)

    # Get the ICA data epoched on the emap saccades
    sac_ica_data = sac_ica_sources.get_data()
    fix_ica_data = fix_ica_sources.get_data()

    # Compute variance along 3rd axis (time)
    sac_variance = np.var(sac_ica_data, axis=2)
    fix_variance = np.var(fix_ica_data, axis=2)

    # Create directory
    fig_path = plot_path + f'ICA/{subject.subject_id}/Ploch/'
    os.makedirs(fig_path, exist_ok=True)

    # Disable displaying figures
    plt.ioff()
    time.sleep(1)
    print('Plotting saccades and fixations variance distributions')
    for n_comp in range(ica.n_components):
        print(f'\rComponent {n_comp}', end='')
        fig = plt.figure()
        plt.hist(sac_variance[:, n_comp], bins=30, alpha=0.6, density=True, label='Saccades')
        plt.hist(fix_variance[:, n_comp], bins=30, alpha=0.6, density=True, label='Fixations')
        plt.legend()
        # Save figure
        save.fig(fig=fig, path=fig_path, fname=f'component_{n_comp}')
    plt.close('all')
    print()
    # Reenable figures
    plt.ion()



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
        ica.plot_properties(meg_downsampled, picks=components, psd_args=dict(fmax=40), show=False)

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

