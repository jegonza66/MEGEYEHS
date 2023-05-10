import mne
import numpy as np
import functions_general
import matplotlib.pyplot as plt
from paths import paths
import os
import time
import save
import load
import setup


def define_events(subject, meg_data, epoch_id, mss=None, trials=None, evt_dur=None, screen=None, tgt=None,
                  dir=None, evt_from_df=False):

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
        if evt_dur:
            metadata = metadata.loc[(metadata['duration'] >= evt_dur)]
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

    else:
        # Get events from annotations
        all_events, all_event_id = mne.events_from_annotations(meg_data, verbose=False)

        # Select epochs
        epoch_keys = [key for key in all_event_id.keys() if epoch_id in key]
        if 'sac' not in epoch_id:
            epoch_keys = [key for key in epoch_keys if 'sac' not in key]
        if 'fix' not in epoch_id:
            epoch_keys = [key for key in epoch_keys if 'fix' not in key]
        if trials != None:
            try:
                epoch_keys = [epoch_key for epoch_key in epoch_keys if
                              (epoch_key.split('_t')[-1].split('_')[0] in trials and 'end' not in epoch_key)]
            except:
                print('Trial selection skipped. Epoch_id does not contain trial number.')

        # Set duration limit
        if evt_dur:
            if 'fix' in epoch_id:
                metadata = subject.fixations
            elif 'sac' in epoch_id:
                metadata = subject.saccades
            metadata = metadata.loc[(metadata['duration'] >= evt_dur)]
            metadata_ids = list(metadata['id'])
            epoch_keys = [key for key in epoch_keys if key in metadata_ids]

        # Get events and ids matchig selection
        metadata, events, events_id = mne.epochs.make_metadata(events=all_events, event_id=all_event_id,
                                                               row_events=epoch_keys, tmin=0, tmax=0,
                                                               sfreq=meg_data.info['sfreq'])

        if 'fix' in epoch_id:
            metadata_sup = subject.fixations
        elif 'sac' in epoch_id:
            metadata_sup = subject.saccades

    return metadata, events, events_id, metadata_sup


def epoch_data(subject, mss, corr_ans, tgt_pres, epoch_id, meg_data, tmin, tmax, trial_dur=None, evt_dur=None, baseline=(None, 0), reject=None,
               save_data=False, epochs_save_path=None, epochs_data_fname=None):
    '''
    :param subject:
    :param mss:
    :param corr_ans:
    :param tgt_pres:
    :param epoch_id:
    :param meg_data:
    :param tmin:
    :param tmax:
    :param baseline: tuple
    Baseline start and end times.
    :param reject: float|str|bool
    Peak to peak amplituyde reject parameter. Use 'subject' for subjects default calculated for short fixation epochs.
     Use False for no rejection. Default to 4e-12 for magnetometers.
    :param save_data:
    :param epochs_save_path:
    :param epochs_data_fname:
    :return:
    '''

    # Sanity check to save data
    if save_data and (not epochs_save_path or not epochs_data_fname):
        raise ValueError('Please provide path and filename to save data. If not, set save_data to false.')

    # Trials
    cond_trials, bh_data_sub = functions_general.get_condition_trials(subject=subject, mss=mss, trial_dur=trial_dur,
                                                                      evt_dur=evt_dur, corr_ans=corr_ans, tgt_pres=tgt_pres)
    # Define events
    metadata, events, events_id, metadata_sup = define_events(subject=subject, epoch_id=epoch_id, evt_dur=evt_dur,
                                                              trials=cond_trials, meg_data=meg_data)
    # Reject based on channel amplitude
    if reject == None:
        # Not setting reject parameter will set to default subject value
        reject = dict(mag=4e-12)
    elif reject == False:
        # Setting reject parameter to False uses No rejection (None in mne will not reject)
        reject = None
    elif reject == 'subject':
        reject = dict(mag=subject.config.general.reject_amp)

    # Epoch data
    epochs = mne.Epochs(raw=meg_data, events=events, event_id=events_id, tmin=tmin, tmax=tmax, reject=reject,
                        event_repeated='drop', metadata=metadata, preload=True, baseline=baseline)
    # Drop bad epochs
    epochs.drop_bad()

    if metadata_sup is not None:
        metadata_sup = metadata_sup.loc[(metadata_sup['id'].isin(epochs.metadata['event_name']))].reset_index(drop=True)
        epochs.metadata = metadata_sup

    if save_data:
        # Save epoched data
        epochs.reset_drop_log_selection()
        os.makedirs(epochs_save_path, exist_ok=True)
        epochs.save(epochs_save_path + epochs_data_fname, overwrite=True)

    return epochs, events


def time_frequency(epochs, l_freq, h_freq, freqs_type, n_cycles_div=4., return_itc=True, save_data=False, trf_save_path=None,
                   power_data_fname=None, itc_data_fname=None):

    # Sanity check to save data
    if save_data and (not trf_save_path or not power_data_fname or not itc_data_fname):
        raise ValueError('Please provide path and filename to save data. Else, set save_data to false.')

    # Compute power over frequencies
    print('Computing power and ITC')
    if freqs_type == 'log':
        freqs = np.logspace(*np.log10([l_freq, h_freq]), num=40)
    elif freqs_type == 'lin':
        freqs = np.linspace(l_freq, h_freq, num=h_freq - l_freq + 1)  # 1 Hz bands
    n_cycles = freqs / n_cycles_div  # different number of cycle per frequency
    power, itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                               return_itc=return_itc, decim=3, n_jobs=None, verbose=True)

    if save_data:
        # Save trf data
        os.makedirs(trf_save_path, exist_ok=True)
        power.save(trf_save_path + power_data_fname, overwrite=True)
        itc.save(trf_save_path + itc_data_fname, overwrite=True)

    return power, itc


def get_plot_tf(tfr, plot_xlim=(None, None), plot_max=True, plot_min=True):
    if plot_xlim:
        tfr_crop = tfr.copy().crop(tmin=plot_xlim[0], tmax=plot_xlim[1])
    else:
        tfr_crop = tfr.copy()

    timefreqs = []

    if plot_max:
        max_ravel = tfr_crop.data.mean(0).argmax()
        freq_idx = int(max_ravel / len(tfr_crop.times))
        time_percent = max_ravel / len(tfr_crop.times) - freq_idx
        time_idx = round(time_percent * len(tfr_crop.times))
        max_timefreq = (tfr_crop.times[time_idx], tfr_crop.freqs[freq_idx])
        timefreqs.append(max_timefreq)

    if plot_min:
        min_ravel = tfr_crop.data.mean(0).argmin()
        freq_idx = int(min_ravel / len(tfr_crop.times))
        time_percent = min_ravel / len(tfr_crop.times) - freq_idx
        time_idx = round(time_percent * len(tfr_crop.times))
        min_timefreq = (tfr_crop.times[time_idx], tfr_crop.freqs[freq_idx])
        timefreqs.append(min_timefreq)

    timefreqs.sort()

    return timefreqs


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


def noise_cov(exp_info, subject, bads, use_ica_data, reject=dict(mag=4e-12), rank=None):
    '''
    Compute background noise covariance matrix for source estimation.
    :param exp_info:
    :param subject:
    :param meg_data:
    :param use_ica_data:
    :return: noise_cov
    '''

    # Define background noise session id
    noise_date_id = exp_info.subjects_noise[subject.subject_id]

    # Load data
    noise = setup.noise(exp_info=exp_info, date_id=noise_date_id)
    raw_noise = noise.load_preproc_data()

    # Set bads to match participant's
    raw_noise.info['bads'] = bads

    if use_ica_data:
        # ICA
        save_path_ica = paths().ica_path() + subject.subject_id + '/'
        ica_fname = 'ICA.pkl'

        # Load ICA
        ica = load.var(file_path=save_path_ica + ica_fname)
        print('ICA object loaded')

        # Get excluded components from subject and apply ICA to background noise
        ica.exclude = subject.ex_components
        # Load raw noise data to apply ICA
        raw_noise.load_data()
        ica.apply(raw_noise)

    # Pick meg channels for source modeling
    raw_noise.pick('meg')

    # Compute covariance to withdraw from meg data
    noise_cov = mne.compute_raw_covariance(raw_noise, reject=reject, rank=rank)

    return noise_cov