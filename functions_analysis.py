import mne
import numpy as np
import functions_general
import matplotlib.pyplot as plt
from paths import paths
import os
import time
import save


def define_events(subject, epoch_id, screen, mss, dur, tgt, dir, meg_data, evt_from_df=False, evt_from_annot=True):

    print('Defining events')

    metadata_sup = None

    if evt_from_df:
        if 'fix' in epoch_id:
            metadata = subject.fixations
        elif 'sac' in epoch_id:
            metadata = subject.saccades

        # Get events from fix/sac Dataframe
        if screen:
            metadata = metadata.loc[(metadata['screen'] == screen)]
        if mss:
            metadata = metadata.loc[(metadata['mss'] == mss)]
        if dur:
            metadata = metadata.loc[(metadata['duration'] >= dur)]
        if 'fix' in epoch_id:
            if tgt == 1:
                metadata = metadata.loc[(metadata['fix_target'] == tgt)]
            elif tgt == 0:
                metadata = metadata.loc[(metadata['fix_target'] == tgt)]
        if 'sac' in epoch_id:
            if dir:
                metadata = metadata.loc[(metadata['dir'] == dir)]

        metadata.reset_index(drop=True, inplace=True)

        events_samples, event_times = functions_general.find_nearest(meg_data.times, metadata['onset'])

        events = np.zeros((len(events_samples), 3)).astype(int)
        events[:, 0] = events_samples
        events[:, 2] = metadata.index

        events_id = dict(zip(metadata.id, metadata.index))

    if evt_from_annot:
        # Get events from annotations
        all_events, all_event_id = mne.events_from_annotations(meg_data, verbose=False)

        # Select epochs
        epoch_keys = [key for key in all_event_id.keys() if epoch_id in key]
        if 'sac' not in epoch_id:
            epoch_keys = [key for key in epoch_keys if 'sac' not in key]
        if 'fix' not in epoch_id:
            epoch_keys = [key for key in epoch_keys if 'fix' not in key]
        if screen:
            epoch_keys = [epoch_key for epoch_key in epoch_keys if f'{screen}' in epoch_key]
        if mss:
            trials_mss = subject.bh_data.loc[subject.bh_data['Nstim'] == mss].index + 1  # add 1 due to python 0th indexing
            epoch_keys = [epoch_key for epoch_key in epoch_keys if int(epoch_key.split('t')[-1]) in trials_mss]

        # Get events and ids matchig selection
        metadata, events, events_id = mne.epochs.make_metadata(events=all_events, event_id=all_event_id,
                                                               row_events=epoch_keys, tmin=0, tmax=0,
                                                               sfreq=meg_data.info['sfreq'])

        if 'fix' in epoch_id:
            metadata_sup = subject.fixations
        elif 'sac' in epoch_id:
            metadata_sup = subject.saccades

    return metadata, events, events_id, metadata_sup


def ocular_components_ploch(subject, meg_downsampled, ica, sac_id='sac_emap', fix_id='fix_emap' , threshold=1.1,
                            plot_distributions=True):
    '''
    Ploch's algorithm for saccadic artifacts detection by variance comparison

    :param subject:
    :param meg_downsampled:
    :param ica:
    :param save_distributions:
    :return: ocular_components
    '''

    # Screen
    screen = functions_general.get_screen(epoch_id=sac_id)
    # MSS
    mss = functions_general.get_mss(epoch_id=sac_id)
    # Item
    tgt = functions_general.get_item(epoch_id=sac_id)
    # Saccades direction
    dir = functions_general.get_dir(epoch_id=sac_id)

    # ica_sources_meg_data = ica.get_sources(meg_downsampled)

    # Define events
    print('Saccades')
    sac_metadata, sac_events, sac_events_id, sac_metadata_sup = \
        define_events(subject=subject, epoch_id=sac_id, screen=screen, mss=mss, dur=None,
                                         tgt=tgt, dir=dir, meg_data=meg_downsampled)

    print('Fixations')
    fix_metadata, fix_events, fix_events_id, fix_metadata_sup = \
        define_events(subject=subject, epoch_id=fix_id, screen=screen, mss=mss, dur=None,
                                         tgt=tgt, dir=dir, meg_data=meg_downsampled)

    # Get time windows from epoch_id name
    sac_tmin = -0.005  # Add previous 5 ms
    sac_tmax = sac_metadata_sup['duration'].mean()
    fix_tmin = 0
    fix_tmax = fix_metadata_sup['duration'].min()

    # Epoch data
    sac_epochs = mne.Epochs(raw=meg_downsampled, events=sac_events, event_id=sac_events_id, tmin=sac_tmin,
                            tmax=sac_tmax, reject=None,
                            event_repeated='drop', metadata=sac_metadata, preload=True, baseline=(0, 0))
    fix_epochs = mne.Epochs(raw=meg_downsampled, events=fix_events, event_id=fix_events_id, tmin=fix_tmin,
                            tmax=fix_tmax, reject=None,
                            event_repeated='drop', metadata=fix_metadata, preload=True, baseline=(0, 0))

    # Append saccades df as epochs metadata
    if sac_metadata_sup is not None:
        sac_metadata_sup = sac_metadata_sup.loc[
            (sac_metadata_sup['id'].isin(sac_epochs.metadata['event_name']))].reset_index(drop=True)
        sac_epochs.metadata = sac_metadata_sup
    if fix_metadata_sup is not None:
        fix_metadata_sup = fix_metadata_sup.loc[
            (fix_metadata_sup['id'].isin(fix_epochs.metadata['event_name']))].reset_index(drop=True)
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

    # Plot components distributions
    if plot_distributions:
        # Create directory
        plot_path = paths().plots_path()
        fig_path = plot_path + f'ICA/{subject.subject_id}/Variance_distributions/'
        os.makedirs(fig_path, exist_ok=True)

        # Disable displaying figures
        plt.ioff()
        time.sleep(1)

        # Plot componets distributions
        print('Plotting saccades and fixations variance distributions')
        for n_comp in range(ica.n_components):
            print(f'\rComponent {n_comp}', end='')
            fig = plt.figure()
            plt.hist(sac_variance[:, n_comp], bins=30, alpha=0.6, density=True, label='Saccades')
            plt.hist(fix_variance[:, n_comp], bins=30, alpha=0.6, density=True, label='Fixations')
            plt.legend()
            plt.title(f'ICA component {n_comp}')

            # Save figure
            save.fig(fig=fig, path=fig_path, fname=f'component_{n_comp}')
            plt.close(fig)
        print()

        # Reenable figures
        plt.ion()

    # Compute mean component variances
    mean_sac_variance = np.mean(sac_variance, axis=0)
    mean_fix_variance = np.mean(fix_variance, axis=0)

    # Compute variance ratio
    variance_ratio = mean_sac_variance / mean_fix_variance

    # Compute artifactual components
    ocular_components = np.where(variance_ratio > threshold)[0]

    print('The ocular components to exclude based on the variance ratio between saccades and fixations with a '
          f'threshold of {threshold} are: {ocular_components}')

    return ocular_components, sac_epochs, fix_epochs