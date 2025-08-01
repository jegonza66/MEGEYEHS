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
from mne.decoding import ReceptiveField
from scipy import stats as stats
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc
from tqdm import tqdm
from mne.stats import permutation_cluster_1samp_test
from sklearn.cluster import KMeans
import umap
import pandas as pd
import seaborn as sns
import plot_general


# ---------- Epoch Data ---------- #
def define_events(subject, meg_data, epoch_id, trial_num=None, evt_dur=None, epoch_keys=None, rel_sac=False):
    '''

    :param subject:
    :param meg_data:
    :param epoch_id:
    :param trials:
    :param evt_dur:
    :param epoch_keys: List of str indicating epoch_ids to epoch data on. Default: None.
    If not provided, will epoch data based on other parameters. If provided will override all other parameters.
    :return:
    '''
    print('Defining events')

    metadata_sup = None
    # Get events from annotations
    all_events, all_event_id = mne.events_from_annotations(meg_data, verbose=False)

    if epoch_keys is None:
        # Define epoch keys as all events
        epoch_keys = []

        # Iterate over posible multiple epoch ids
        for epoch_sub_id in epoch_id.split('+'):

            # Select epochs
            epoch_keys += [key for key in all_event_id if epoch_sub_id in key]
            if 'sac' not in epoch_sub_id:
                epoch_keys = [key for key in epoch_keys if 'sac' not in key]
            if 'fix' not in epoch_sub_id:
                epoch_keys = [key for key in epoch_keys if 'fix' not in key]
            if trial_num != None and any('_t' in epoch_key for epoch_key in epoch_keys):
                try:
                    if 'vsend' in epoch_sub_id or 'cross1' in epoch_sub_id:
                        epoch_keys = [epoch_key for epoch_key in epoch_keys if epoch_key.split('_t')[-1] in trial_num]
                    else:
                        epoch_keys = [epoch_key for epoch_key in epoch_keys if (epoch_key.split('_t')[-1].split('_')[0] in trial_num and 'end' not in epoch_key)]
                except:
                    print('Trial selection skipped. Epoch_id does not contain trial number.')

            if len(epoch_keys) == 0:
                raise ValueError('No valid epoch_ids provided.')

            # Set duration limit
            if 'fix' in epoch_sub_id:
                metadata = subject.fixations
                if evt_dur:
                    if evt_dur >= 0:
                        metadata = metadata.loc[(metadata['duration'] >= evt_dur)]
                        metadata_ids = list(metadata['id'])
                        epoch_keys = [key for key in epoch_keys if key in metadata_ids]
                    elif evt_dur < 0:
                        metadata = metadata.loc[(metadata['duration'] <= abs(evt_dur))]
                        metadata_ids = list(metadata['id'])
                        epoch_keys = [key for key in epoch_keys if key in metadata_ids]

            elif 'sac' in epoch_sub_id:
                metadata = subject.saccades
                if evt_dur:
                    if evt_dur >= 0:
                        metadata = metadata.loc[(metadata['duration'] >= evt_dur)]
                        metadata_ids = list(metadata['id'])
                        epoch_keys = [key for key in epoch_keys if key in metadata_ids]
                    elif evt_dur < 0:
                        metadata = metadata.loc[(metadata['duration'] <= abs(evt_dur))]
                        metadata_ids = list(metadata['id'])
                        epoch_keys = [key for key in epoch_keys if key in metadata_ids]

            # Change fixations for previous saccades
            if rel_sac:
                epoch_keys_idx = metadata.loc[metadata['id'].isin(epoch_keys), f'{rel_sac}_sac']
                epoch_keys_idx.dropna(inplace=True)
                epoch_keys = subject.saccades.loc[epoch_keys_idx, 'id']
                epoch_keys = epoch_keys[epoch_keys != 'None'].to_list()

    # Get events and ids matching selection
    metadata, events, events_id = mne.epochs.make_metadata(events=all_events, event_id=all_event_id, row_events=epoch_keys, tmin=0, tmax=0, sfreq=meg_data.info['sfreq'])

    if 'fix' in epoch_id and not rel_sac:
        metadata_sup = subject.fixations
    elif 'sac' in epoch_id:
        metadata_sup = subject.saccades

    return metadata, events, events_id, metadata_sup


def epoch_data(subject, mss, corr_ans, tgt_pres, epoch_id, meg_data, tmin, tmax, trial_num=None, trial_dur=None, evt_dur=None, rel_sac=False,
               baseline=(None, 0), reject=None, save_data=False, epochs_save_path=None, epochs_data_fname=None):
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
    if not trial_num:
        trial_num, bh_data_sub = functions_general.get_condition_trials(subject=subject, mss=mss, trial_dur=trial_dur, corr_ans=corr_ans, tgt_pres=tgt_pres)
    elif isinstance(trial_num, int):
        trial_num = [str(trial_num)]

    # Redefine epoch id
    if rel_sac and 'sac' in epoch_id:
        fix_epoch_id = epoch_id.replace('sac', 'fix')

        # If running subsampled rel saccades epochs, try to load the fixations data to extract relative saccades from it. If no defined subsampled fixations epochs, run subsampling script
        try:
            # load fixations epochs (in case subsampled)
            # Get time windows from epoch_id name
            fix_tmin, fix_tmax, _ = functions_general.get_time_lims(epoch_id=fix_epoch_id, mss=mss)

            # Windows durations
            cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()
            # Get baseline duration for epoch_id
            fix_bline, _ = functions_general.get_baseline_duration(epoch_id=fix_epoch_id, mss=mss, tmin=tmin, tmax=tmax, cross1_dur=cross1_dur, mss_duration=mss_duration,
                                                                   cross2_dur=cross2_dur)

            # Correct epochs path for fixations
            fix_epochs_save_path = epochs_save_path.replace(f'{rel_sac}_', '').replace('sac', 'fix')
            fix_epochs_save_path = fix_epochs_save_path.replace(f'{tmin}_{tmax}', f'{fix_tmin}_{fix_tmax}').replace(fix_epochs_save_path.split('bline')[-1], f'{fix_bline}/')

            # Load epochs
            fix_epochs = mne.read_epochs(fix_epochs_save_path + epochs_data_fname)

            # Extract rel sac epoch keys from metadata
            fix_metadata = fix_epochs.metadata
            epoch_keys_idx = fix_metadata[f'{rel_sac}_sac']
            epoch_keys_idx.dropna(inplace=True)
            epoch_keys = subject.saccades.loc[epoch_keys_idx, 'id']
            epoch_keys = epoch_keys[epoch_keys != 'None'].to_list()

            # Get events from annotations
            all_events, all_event_id = mne.events_from_annotations(meg_data, verbose=False)

            # Get events and ids matching selection
            metadata, events, events_id = mne.epochs.make_metadata(events=all_events, event_id=all_event_id, row_events=epoch_keys, tmin=0, tmax=0,
                                                                   sfreq=meg_data.info['sfreq'])
            metadata_sup = subject.saccades
        except:

            # Define events
            metadata, events, events_id, metadata_sup = define_events(subject=subject, meg_data=meg_data, epoch_id=fix_epoch_id, evt_dur=evt_dur, trial_num=trial_num, rel_sac=rel_sac)

    else:
        # Define events
        metadata, events, events_id, metadata_sup = define_events(subject=subject, meg_data=meg_data, epoch_id=epoch_id, evt_dur=evt_dur, trial_num=trial_num,
                                                                  rel_sac=rel_sac)

    # Reject based on channel amplitude
    if reject == False:
        # Setting reject parameter to False uses No rejection (None in mne will not reject)
        reject = None
    elif reject == 'subject':
        reject = dict(mag=subject.config.general.reject_amp)
    elif reject == None:
        reject = dict(mag=5e-12)

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


# ---------- Time Frequency analysis ---------- #
def time_frequency(epochs, l_freq, h_freq, freqs_type='lin', n_cycles_div=2., average=True, return_itc=True, output='power',
                   save_data=False, trf_save_path=None, power_data_fname=None, itc_data_fname=None, n_jobs=4):

    # Sanity check to save data
    if save_data and (not trf_save_path or not power_data_fname) or (return_itc and not itc_data_fname):
        raise ValueError('Please provide path and filename to save data. Else, set save_data to false.')

    # Compute power over frequencies
    print('Computing power and ITC')
    if freqs_type == 'log':
        freqs = np.logspace(np.log10([l_freq, h_freq]), num=40)
    elif freqs_type == 'lin':
        freqs = np.linspace(l_freq, h_freq, num=h_freq - l_freq + 1)  # 1 Hz bands
    n_cycles = freqs / n_cycles_div  # different number of cycle per frequency

    # Phase output
    if output == 'phase':
        output = 'complex'
        average = False

        power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=average,
                                              output=output, return_itc=return_itc, decim=3, n_jobs=n_jobs, verbose=True)
        power.data = np.imag(power.data)
        power = power.average()
        return power

    else:
        if return_itc:
            power, itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                                       average=average, output=output,
                                                       return_itc=return_itc, decim=3, n_jobs=n_jobs, verbose=True)
            if save_data:
                # Save trf data
                os.makedirs(trf_save_path, exist_ok=True)
                power.save(trf_save_path + power_data_fname, overwrite=True)
                itc.save(trf_save_path + itc_data_fname, overwrite=True)

            return power, itc

        else:
            power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=average,
                                                  output=output, return_itc=return_itc, decim=3, n_jobs=n_jobs, verbose=True)

            if save_data:
                # Save trf data
                os.makedirs(trf_save_path, exist_ok=True)
                power.save(trf_save_path + power_data_fname, overwrite=True)

            return power


def time_frequency_multitaper(epochs, l_freq, h_freq, freqs_type='lin', n_cycles_div=2., average=True, return_itc=True,
                               save_data=False, trf_save_path=None, power_data_fname=None, itc_data_fname=None, n_jobs=4):

    # Sanity check to save data
    if save_data and (not trf_save_path or not power_data_fname) or (return_itc and not itc_data_fname):
        raise ValueError('Please provide path and filename to save data. Else, set save_data to false.')

    # Compute power over frequencies
    print('Computing power and ITC')
    if freqs_type == 'log':
        freqs = np.logspace(np.log10([l_freq, h_freq]), num=40)
    elif freqs_type == 'lin':
        freqs = np.linspace(l_freq, h_freq, num=h_freq - l_freq + 1)  # 1 Hz bands
    n_cycles = freqs / n_cycles_div  # different number of cycle per frequency
    if return_itc:
        power, itc = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                                   average=average,
                                                   return_itc=return_itc, decim=3, n_jobs=n_jobs, verbose=True)
        if save_data:
            # Save trf data
            os.makedirs(trf_save_path, exist_ok=True)
            power.save(trf_save_path + power_data_fname, overwrite=True)
            itc.save(trf_save_path + itc_data_fname, overwrite=True)

        return power, itc

    else:
        power = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, average=average,
                                              return_itc=return_itc, decim=3, n_jobs=n_jobs, verbose=True)
        if save_data:
            # Save trf data
            os.makedirs(trf_save_path, exist_ok=True)
            power.save(trf_save_path + power_data_fname, overwrite=True)

        return power


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


# ---------- ICA ocular artifacts ---------- #
def ocular_components_ploch(subject, meg_downsampled, ica, sac_id='sac_emap', fix_id='fix_emap' , reject={'mag': 5e-12}, threshold=1.1,
                            plot_distributions=True):
    '''
    Ploch's algorithm for saccadic artifacts detection by variance comparison

    :param subject:
    :param meg_downsampled:
    :param ica:
    :param save_distributions:
    :return: ocular_components
    '''

    # Define events
    print('Saccades')
    sac_metadata, sac_events, sac_events_id, sac_metadata_sup = \
        define_events(subject=subject, epoch_id=sac_id, meg_data=meg_downsampled)

    print('Fixations')
    fix_metadata, fix_events, fix_events_id, fix_metadata_sup = \
        define_events(subject=subject, epoch_id=fix_id, meg_data=meg_downsampled)

    # Get time windows from epoch_id name
    sac_tmin = -0.005  # Add previous 5 ms
    sac_tmax = sac_metadata_sup['duration'].mean()
    fix_tmin = 0
    fix_tmax = fix_metadata_sup['duration'].min()

    # Epoch data
    sac_epochs = mne.Epochs(raw=meg_downsampled, events=sac_events, event_id=sac_events_id, tmin=sac_tmin,
                            tmax=sac_tmax, reject=reject,
                            event_repeated='drop', metadata=sac_metadata, preload=True, baseline=(0, 0))
    fix_epochs = mne.Epochs(raw=meg_downsampled, events=fix_events, event_id=fix_events_id, tmin=fix_tmin,
                            tmax=fix_tmax, reject=reject,
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


#---------- MTRF -----------#
def get_bad_annot_array(meg_data, subj_path, fname, save_var=True):
    # Get bad annotations times
    bad_annotations_idx = [i for i, annot in enumerate(meg_data.annotations.description) if
                           ('bad' in annot or 'BAD' in annot)]
    bad_annotations_time = meg_data.annotations.onset[bad_annotations_idx]
    bad_annotations_duration = meg_data.annotations.duration[bad_annotations_idx]
    bad_annotations_endtime = bad_annotations_time + bad_annotations_duration

    bad_indexes = []
    for i in range(len(bad_annotations_time)):
        bad_annotation_span_idx = np.where(
            np.logical_and((meg_data.times > bad_annotations_time[i]), (meg_data.times < bad_annotations_endtime[i])))[
            0]
        bad_indexes.append(bad_annotation_span_idx)

    # Flatten all indexes and convert to array
    bad_indexes = functions_general.flatten_list(bad_indexes)
    bad_indexes = np.array(bad_indexes)

    # Make bad annotations binary array
    bad_annotations_array = np.ones(len(meg_data.times))
    bad_annotations_array[bad_indexes] = 0

    # Save arrays
    if save_var:
        save.var(var=bad_annotations_array, path=subj_path, fname=fname)

    return bad_annotations_array


def make_mtrf_input(input_arrays, var_name, subject, meg_data, evt_dur, cond_trials, epoch_keys, bad_annotations_array,
                    subj_path, fname, save_var=True):

    # Define events
    metadata, events, _, _ = define_events(subject=subject, epoch_id=var_name, evt_dur=evt_dur, trial_num=cond_trials, meg_data=meg_data,  epoch_keys=epoch_keys)
    # Make input arrays as 0
    input_array = np.zeros(len(meg_data.times))
    # Get events samples index
    evt_idxs = events[:, 0]
    # Set those indexes as 1
    input_array[evt_idxs] = 1
    # Exclude bad annotations
    input_array = input_array * bad_annotations_array
    # Save to all input arrays dictionary
    input_arrays[var_name] = input_array

    # Save arrays
    if save_var:
        save.var(var=input_array, path=subj_path, fname=fname)

    return input_arrays


def fit_mtrf(meg_data, tmin, tmax, model_input, chs_id, standarize=True, fit_power=False, alpha=0, n_jobs=4):

    # Define mTRF model
    rf = ReceptiveField(tmin, tmax, meg_data.info['sfreq'], estimator=alpha, scoring='corrcoef', verbose=False, n_jobs=n_jobs)

    # Get subset channels data as array
    picks = functions_general.pick_chs(chs_id=chs_id, info=meg_data.info)
    meg_sub = meg_data.copy().pick(picks)

    # Apply hilbert and extract envelope
    if fit_power:
        meg_sub = meg_sub.apply_hilbert(envelope=True)

    meg_data_array = meg_sub.get_data()

    if standarize:
        # Standarize data
        print('Computing z-score...')
        meg_data_array = np.expand_dims(meg_data_array, axis=0)  # Need shape (n_epochs, n_channels n_times)
        meg_data_array = mne.decoding.Scaler(info=meg_sub.info, scalings='mean').fit_transform(meg_data_array)
        meg_data_array = meg_data_array.squeeze()
    # Transpose to input the model
    meg_data_array = meg_data_array.T

    # Fit TRF
    rf.fit(model_input, meg_data_array)

    return rf


def compute_trf(subject, meg_data, trial_params, trf_params, meg_params, all_chs_regions=['frontal', 'temporal', 'central', 'parietal', 'occipital'],
                save_data=False, trf_path=None, trf_fname=None):

    print(f"Computing TRF for {trf_params['input_features']}")

    # Get condition trials
    cond_trials, bh_data_sub = functions_general.get_condition_trials(subject=subject, mss=trial_params['mss'], trial_dur=trial_params['trialdur'],
                                                                      corr_ans=trial_params['corrans'], tgt_pres=trial_params['tgtpres'])

    # Bad annotations filepath
    subj_path = paths().save_path() + f'TRF/{subject.subject_id}/'
    fname_bad_annot = f'bad_annot_array.pkl'

    try:
        bad_annotations_array = load.var(subj_path + fname_bad_annot)
        print(f'Loaded bad annotations array')
    except:
        print(f'Computing bad annotations array...')
        bad_annotations_array = get_bad_annot_array(meg_data=meg_data, subj_path=subj_path, fname=fname_bad_annot)

    # Iterate over input features
    input_arrays = {}
    for feature in trf_params['input_features']:

        subj_path = paths().save_path() + f'TRF/{subject.subject_id}/'
        fname_var = (f"{feature}_mss{trial_params['mss']}_corrans{trial_params['corrans']}_tgtpres{trial_params['tgtpres']}_trialdur{trial_params['trialdur']}_"
                     f"evtdur{trial_params['evtdur']}_array.pkl")

        try:
            input_arrays[feature] = load.var(file_path=subj_path + fname_var)
            print(f"Loaded input array for {feature}")

        except:
            print(f'Computing input array for {feature}...')
            # Exception for subsampled distractor fixations
            if 'sub' in feature:
                # Subsampled epochs path
                epochs_save_id = (f"{feature}_mss{trial_params['mss']}_corrans{trial_params['corrans']}_tgtpres{trial_params['tgtpres']}_"
                                  f"trialdur{trial_params['trialdur']}_evtdur{trial_params['evtdur']}")
                epochs_save_path = paths().save_path() + (f"Epochs_{meg_params['data_type']}/Band_{meg_params['band_id']}/{epochs_save_id}_{trf_params['tmin']}_"
                                                          f"{trf_params['tmax']}_bline{trf_params['baseline']}/")

                epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'

                # Load epoched data
                epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)

                # Get epochs id from metadata
                epoch_keys = epochs.metadata['id'].to_list()

            else:
                epoch_keys = None

            input_arrays = make_mtrf_input(input_arrays=input_arrays, var_name=feature,
                                           subject=subject, meg_data=meg_data, evt_dur=trial_params['evtdur'],
                                           cond_trials=cond_trials, epoch_keys=epoch_keys,
                                           bad_annotations_array=bad_annotations_array,
                                           subj_path=subj_path, fname=fname_var)

    # Concatenate input arrays as one
    model_input = np.array([input_arrays[key] for key in input_arrays.keys()]).T

    # All regions or selected (multiple) regions
    if meg_params['chs_id'] == 'mag' or '_' in meg_params['chs_id']:
        # rf as a dictionary containing the rf of each region
        rf = {}
        # iterate over regions
        for chs_subset in all_chs_regions:
            # Use only regions in channels id, or all in case of chs_id == 'mag'
            if chs_subset in meg_params['chs_id'] or meg_params['chs_id'] == 'mag':
                print(f'Fitting mTRF for region {chs_subset}')
                rf[chs_subset] = fit_mtrf(meg_data=meg_data, tmin=trf_params['tmin'], tmax=trf_params['tmax'], alpha=trf_params['alpha'], fit_power=trf_params['fit_power'],
                                                             model_input=model_input, chs_id=chs_subset, standarize=trf_params['standarize'], n_jobs=4)
    # One region
    else:
        rf = fit_mtrf(meg_data=meg_data, tmin=trf_params['tmin'], tmax=trf_params['tmax'], alpha=trf_params['alpha'], fit_power=trf_params['fit_power'],
                                                             model_input=model_input, chs_id=meg_params['chs_id'], standarize=trf_params['standarize'], n_jobs=4)
    # Save TRF
    if save_data:
        save.var(var=rf, path=trf_path, fname=trf_fname)

    return rf


def make_trf_evoked(subject, rf, meg_data, trf_params, meg_params, evokeds=None, display_figs=False, plot_individuals=True, save_fig=True, fig_path=None):
    """
    Get model coeficients as separate responses to each feature.

    Parameters
    ----------
    subject
    rf
    meg_data
    evokeds
    trf_params
    trial_params
    meg_params
    display_figs
    plot_individuals
    save_fig
    fig_path

    Returns
    -------

    """

    # Sanity check
    if save_fig and not fig_path:
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    # Variables to store coefficients, and "evoked" data of the subject.
    trf = {}
    subj_evoked = {}
    subj_evoked_list = {}
    for i, feature in enumerate(trf_params['input_features']):

        # All or multiple regions
        if meg_params['chs_id'] == 'mag' or '_' in meg_params['chs_id']:

            # Define evoked from TRF list to concatenate all
            subj_evoked_list[feature] = []

            # iterate over regions
            for chs_idx, chs_subset in enumerate(rf.keys()):
                # Get channels subset info
                picks = functions_general.pick_chs(chs_id=chs_subset, info=meg_data.info)
                meg_sub = meg_data.copy().pick(picks)

                # Get TRF coeficients from chs subset
                trf[feature] = rf[chs_subset].coef_[:, i, :]

                if chs_idx == 0:
                    # Define evoked object from arrays of TRF
                    subj_evoked[feature] = mne.EvokedArray(data=trf[feature], info=meg_sub.info, tmin=trf_params['tmin'], baseline=trf_params['baseline'])
                else:
                    # Append evoked object from arrays of TRF to list, to concatenate all
                    subj_evoked_list[feature].append(mne.EvokedArray(data=trf[feature], info=meg_sub.info, tmin=trf_params['tmin'], baseline=trf_params['baseline']))

            # Concatenate evoked from al regions
            subj_evoked[feature] = subj_evoked[feature].add_channels(subj_evoked_list[feature])

        else:
            trf[feature] = rf.coef_[:, i, :]
            # Define evoked objects from arrays of TRF
            subj_evoked[feature] = mne.EvokedArray(data=trf[feature], info=meg_sub.info, tmin=trf_params['tmin'], baseline=trf_params['baseline'])

        # Append for Grand average
        if evokeds != None:
            evokeds[feature].append(subj_evoked[feature])

        # Plot
        if plot_individuals:
            fig = subj_evoked[feature].plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(trf_params['tmin'], trf_params['tmax']), titles=feature)
            fig.suptitle(f'{feature}')

        if save_fig:
            # Save
            fig_path_subj = fig_path + f'{subject.subject_id}/'
            fname = f"{feature}_{meg_params['chs_id']}"
            save.fig(fig=fig, fname=fname, path=fig_path_subj)

    return subj_evoked, evokeds


def trf_grand_average(feature_evokeds, trf_params, trial_params, meg_params, display_figs=False, save_fig=True, fig_path=None):

    # Sanity check
    if save_fig and not fig_path:
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    grand_avg = {}
    for feature in trf_params['input_features']:
        # Compute grand average
        grand_avg[feature] = mne.grand_average(feature_evokeds[feature], interpolate_bads=True)
        plot_times_idx = np.where((grand_avg[feature].times > trf_params['tmin']) & (grand_avg[feature].times < trf_params['tmax']))[0]
        data = grand_avg[feature].get_data()[:, plot_times_idx]

        # Plot
        fig = grand_avg[feature].plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(trf_params['tmin'], trf_params['tmax']), titles=feature)

        if save_fig:
            # Save
            fname = f"{feature}_GA_{meg_params['chs_id']}"
            save.fig(fig=fig, fname=fname, path=fig_path)

    return grand_avg


def reconstruct_meg_from_trf(subject, rf, meg_data,  picks,  trial_params, trf_params, meg_params):
    # Get TRF from all regions and features
    rf_data = {}
    for i, feature in enumerate(trf_params['input_features']):
        chs_idx = 0
        rf_data[feature] = np.zeros((len(picks), len(rf[list(rf.keys())[0]].delays_)))
        for region in meg_params['chs_id'].split('_'):
            rf_data[feature][chs_idx: chs_idx + rf[region].coef_.shape[0]] = rf[region].coef_[:, i, :]
            chs_idx += rf[region].coef_.shape[0]

    # Negative and positive time samples
    neg_delays_len = -rf[list(rf.keys())[0]].delays_[0]
    pos_delays_len = rf[list(rf.keys())[0]].delays_[-1] + 1  # To account for t=0

    # Reconstructed signal variable
    meg_sub = meg_data.copy().pick(picks)
    reconstructed_data = np.zeros((len(picks), len(meg_data.times)))

    for i, feature in enumerate(trf_params['input_features']):

        subj_path = paths().save_path() + f'TRF/{subject.subject_id}/'
        fname_var = (f"{feature}_mss{trial_params['mss']}_corrans{trial_params['corrans']}_tgtpres{trial_params['tgtpres']}_trialdur{trial_params['trialdur']}_"
                     f"evtdur{trial_params['evtdur']}_array.pkl")

        input_feature = load.var(file_path=subj_path + fname_var)

        print(f'Reconstructing signal from {feature}')
        for event in tqdm(np.where(input_feature == 1)[0]):
            reconstructed_data[:, event - neg_delays_len: event + pos_delays_len] += rf_data[feature]

    reconstructed_meg = mne.io.RawArray(reconstructed_data, meg_sub.info)
    reconstructed_meg.set_annotations(meg_sub.annotations)

    return reconstructed_meg


#----- Statistical tests -----#

# ---------- Source reconstruction ---------- #
def noise_cov(exp_info, subject, bads, use_ica_data, reject=dict(mag=4e-12), rank=None, high_freq=False):
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

    # Filter in high frequencies
    if high_freq:
        l_freq, h_freq = functions_general.get_freq_band(band_id='HGamma')
        raw_noise = raw_noise.filter(l_freq=l_freq, h_freq=h_freq)

    # Compute covariance to withdraw from meg data
    noise_cov = mne.compute_raw_covariance(raw_noise, reject=reject, rank=rank)

    return noise_cov


def noise_csd(exp_info, subject, bads, use_ica_data, freqs):
    '''
    Compute background noise csd for source estimation.
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
    raw_noise.pick('mag')

    # Compute covariance to withdraw from meg data
    noise_epoch = mne.Epochs(raw_noise, events=np.array([[0, 0, 0]]), tmin=0, tmax=raw_noise.times[-1], baseline=None, preload=True)
    noise_csd = mne.time_frequency.csd_morlet(epochs=noise_epoch, frequencies=freqs)

    return noise_csd


def estimate_sources_cov(subject, meg_params, trial_params, filters, active_times, rank, bline_mode_subj, save_data, cov_save_path, cov_act_fname,
                         cov_baseline_fname, epochs_save_path, epochs_data_fname):

    try:
        # Load covariance matrix
        baseline_cov = mne.read_cov(fname=cov_save_path + cov_baseline_fname)
    except:
        # Load epochs
        try:
            epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
        except:
            # Load MEG
            meg_data = load.meg(subject=subject, meg_params=meg_params, save_data=save_data)

            # Epoch data
            epochs, events = epoch_data(subject=subject, meg_data=meg_data, mss=trial_params['mss'], corr_ans=trial_params['corrans'], tgt_pres=trial_params['tgtpres'],
                                        epoch_id=trial_params['epoch_id'], tmin=trial_params['tmin'], trial_dur=trial_params['trialdur'],
                                        tmax=trial_params['tmax'], reject=trial_params['reject'], baseline=trial_params['baseline'],
                                        save_data=save_data, epochs_save_path=epochs_save_path,
                                        epochs_data_fname=epochs_data_fname)

        # Compute covariance matrices
        baseline_cov = mne.cov.compute_covariance(epochs=epochs, tmin=trial_params['baseline'][0], tmax=trial_params['baseline'][1], method="shrunk", rank=dict(mag=rank))
        # Save
        if save_data:
            os.makedirs(cov_save_path, exist_ok=True)
            baseline_cov.save(fname=cov_save_path + cov_baseline_fname, overwrite=True)

    try:
        # Load covariance matrix
        active_cov = mne.read_cov(fname=cov_save_path + cov_act_fname)
    except:
        # Load epochs
        try:
            epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
        except:
            # Compute epochs
            if meg_params['data_type'] == 'ICA':
                if meg_params['band_id'] and meg_params['filter_sensors']:
                    meg_data = load.filtered_data(subject=subject, band_id=meg_params['band_id'], save_data=save_data,
                                                  method=meg_params['filter_method'])
                else:
                    meg_data = load.ica_data(subject=subject)
            elif meg_params['data_type'] == 'RAW':
                if meg_params['band_id'] and meg_params['filter_sensors']:
                    meg_data = load.filtered_data(subject=subject, band_id=meg_params['band_id'], use_ica_data=False,
                                                  save_data=save_data, method=meg_params['filter_method'])
                else:
                    meg_data = subject.load_preproc_meg_data()

            # Epoch data
            epochs, events = epoch_data(subject=subject, meg_data=meg_data, mss=trial_params['mss'], corr_ans=trial_params['corrans'], tgt_pres=trial_params['tgtpres'],
                                        epoch_id=trial_params['epoch_id'], tmin=trial_params['tmin'], trial_dur=trial_params['trialdur'],
                                        tmax=trial_params['tmax'], reject=trial_params['reject'], baseline=trial_params['baseline'],
                                        save_data=save_data, epochs_save_path=epochs_save_path,
                                        epochs_data_fname=epochs_data_fname)

        # Compute covariance matrices
        active_cov = mne.cov.compute_covariance(epochs=epochs, tmin=active_times[0], tmax=active_times[1], method="shrunk", rank=dict(mag=rank))
        # Save
        if save_data:
            os.makedirs(cov_save_path, exist_ok=True)
            active_cov.save(fname=cov_save_path + cov_act_fname, overwrite=True)

    # Compute sources and apply baseline
    stc_base = mne.beamformer.apply_lcmv_cov(baseline_cov, filters)
    stc_act = mne.beamformer.apply_lcmv_cov(active_cov, filters)

    if bline_mode_subj == 'mean':
        stc = stc_act - stc_base
    elif bline_mode_subj == 'ratio':
        stc = stc_act / stc_base
    elif bline_mode_subj == 'db':
        stc = stc_act / stc_base
        stc.data = 10 * np.log10(stc.data)
    else:
        stc = stc_act

    return stc



def run_source_permutations_test(src, stc, source_data, subject, exp_info, save_regions, fig_path, surf_vol, p_threshold=0.05, n_permutations=1024, desired_tval='TFCE',
                                 mask_negatives=False):

    # Return variables
    stc_all_cluster_vis, significant_voxels, significance_mask, t_thresh_name, time_label = None, None, None, None, None

    # Compute source space adjacency matrix
    print("Computing adjacency matrix")
    adjacency_matrix = mne.spatial_src_adjacency(src)

    # Transpose source_fs_data from shape (subjects x space x time) to shape (subjects x time x space)
    source_data_default = source_data.swapaxes(1, 2)

    # Define the t-value threshold for cluster formation
    if desired_tval == 'TFCE':
        t_thresh = dict(start=0, step=0.1)
    else:
        df = len(exp_info.subjects_ids) - 1  # degrees of freedom for the test
        t_thresh = stats.distributions.t.ppf(1 - desired_tval / 2, df=df)

    # Run permutations
    T_obs, clusters, cluster_p_values, H0 = clu = spatio_temporal_cluster_1samp_test(X=source_data_default,
                                                                                     n_permutations=n_permutations,
                                                                                     adjacency=adjacency_matrix,
                                                                                     n_jobs=4, threshold=t_thresh)

    # Select the clusters that are statistically significant at p
    good_clusters_idx = np.where(cluster_p_values < p_threshold)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]
    significant_pvalues = [cluster_p_values[idx] for idx in good_clusters_idx]

    if len(good_clusters):

        # variable for figure fnames and p_values as title
        if type(t_thresh) == dict:
            time_label = f'{np.round(np.mean(significant_pvalues), 4)} +- {np.round(np.std(significant_pvalues), 4)}'
            t_thresh_name = 'TFCE'
        else:
            time_label = str(significant_pvalues)
            t_thresh_name = round(t_thresh, 2)

        # Get vertices from source space
        fsave_vertices = [s["vertno"] for s in src]

        # Select clusters for visualization
        stc_all_cluster_vis = summarize_clusters_stc(clu=clu, p_thresh=p_threshold, tstep=stc.tstep, vertices=fsave_vertices, subject=subject)

        # Get significant clusters
        significance_mask = np.where(stc_all_cluster_vis.data[:, 0] == 0)[0]
        significant_voxels = np.where(stc_all_cluster_vis.data[:, 0] != 0)[0]

        # Get significant AAL and brodmann regions from mni space
        if save_regions:
            os.makedirs(fig_path, exist_ok=True)
            significant_regions_df = functions_general.get_regions_from_mni(src_default=src, significant_voxels=significant_voxels, save_path=fig_path, surf_vol=surf_vol,
                                                                            t_thresh_name=t_thresh_name, p_threshold=p_threshold, masked_negatves=mask_negatives)

    return stc_all_cluster_vis, significant_voxels, significance_mask, t_thresh_name, time_label, p_threshold


def run_time_frequency_test(data, pval_threshold, t_thresh, min_sig_chs=0, n_permutations=1024):

    # Clusters out type
    if type(t_thresh) == dict:
        out_type = 'indices'
    else:
        out_type = 'mask'

    significant_pvalues = None

    # Permutations cluster test (TFCE if t_thresh as dict)
    t_tfce, clusters, p_tfce, H0 = permutation_cluster_1samp_test(X=data, threshold=t_thresh, n_permutations=n_permutations,
                                                                  out_type=out_type, n_jobs=4)

    # Make clusters mask
    if type(t_thresh) == dict:
        # If TFCE use p-vaues of voxels directly
        p_tfce = p_tfce.reshape(data.shape[-2:])  # Reshape to data's shape
        clusters_mask_plot = p_tfce < pval_threshold
        clusters_mask = None

    else:
        # Get significant clusters
        good_clusters_idx = np.where(p_tfce < pval_threshold)[0]
        significant_clusters = [clusters[idx] for idx in good_clusters_idx]
        significant_pvalues = [p_tfce[idx] for idx in good_clusters_idx]

        # Reshape to data's shape by adding all clusters into one bool array
        clusters_mask = np.zeros(data[0].shape)
        if len(significant_clusters):
            for significant_cluster in significant_clusters:
                clusters_mask += significant_cluster

            if min_sig_chs:
                clusters_mask_plot = clusters_mask.sum(axis=-1) > min_sig_chs
            else:
                clusters_mask_plot = clusters_mask.sum(axis=-1)
            clusters_mask_plot = clusters_mask_plot.astype(bool)

        else:
            clusters_mask_plot = None

    return clusters_mask, clusters_mask_plot, significant_pvalues


def get_labels(subject_code, parcelation, subjects_dir, surf_vol='surface'):

    # Get parcelation labels
    if surf_vol == 'surface':  # or surf_vol == 'mixed':
        # Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
        labels = mne.read_labels_from_annot(subject=subject_code, parc=parcelation, subjects_dir=subjects_dir)
        # Remove 'unknown' label for fsaverage aparc labels
        if 'aparc' in parcelation:
            print("Dropping extra 'unkown' label from lh.")
            drop_idxs = [i for i, label in enumerate(labels) if 'unknown' in label.name.lower()]
            for drop_idx in drop_idxs[::-1]:
                labels.pop(drop_idx)

    if surf_vol == 'volume':
        labels_fname = subjects_dir + f'/{subject_code}/mri/aparc+aseg.mgz'
        labels = mne.get_volume_labels_from_aseg(labels_fname, return_colors=True)
        # Drop extra labels in fsaverage
        drop_idxs = [i for i, label in enumerate(labels[0]) if (label == 'ctx-lh-corpuscallosum' or label == 'ctx-rh-corpuscallosum')]
        for drop_idx in drop_idxs[::-1]:
            labels[0].pop(drop_idx)
            labels[1].pop(drop_idx)

    return labels


def cluster_regions(n_clusters, sig_data, sig_regions, sig_tfr, clusters_masks, l_freq, h_freq, active_times, subjects_dir, display_figs, save_fig, fig_path_diff):
    # Cluster regions
    X = np.array(sig_data)
    final_n_clusters = min(n_clusters, len(sig_regions))

    # Run clustering
    kmeans = KMeans(n_clusters=final_n_clusters, random_state=0, n_init="auto").fit(X)

    # Lower data dimensionality
    X_2d = umap.UMAP(random_state=42).fit_transform(X)

    # Visualize clustering
    clusters = pd.DataFrame({'x': X_2d[:, 0], 'y': X_2d[:, 1], 'c': kmeans.labels_})
    fname = f'Clusters_{final_n_clusters}'

    cluster_fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=clusters, x='x', y='y', hue='c', ax=ax)
    ax.legend(bbox_to_anchor=(1., 1), loc='upper left')
    ax.set_title(fname)
    cluster_fig.tight_layout()

    if save_fig:
        save.fig(fig=cluster_fig, path=fig_path_diff, fname=fname)

    for cluster in np.unique(kmeans.labels_):

        # Close mne 3d figures
        try:
            mne.viz.close_all_3d_figures()
        except:
            pass

        # Define brain for plotting
        Brain = mne.viz.get_brain_class()
        brain = Brain("fsaverage", hemi="split", surf="pial", views=['lat', 'med'], subjects_dir=subjects_dir, size=(1080, 720))

        # Get clusters regions
        cluster_regions_idx = np.where(kmeans.labels_ == cluster)[0]
        # Get cluster's regions data
        sig_cluster_regions = [sig_regions[idx] for idx in cluster_regions_idx]
        sig_cluster_tfr = [sig_tfr[idx] for idx in cluster_regions_idx]
        sig_tf_clusters = [clusters_masks[idx] for idx in cluster_regions_idx]

        # Iterate over clsuter's regions
        for region, tfr, plot_mask in zip(sig_cluster_regions, sig_cluster_tfr, sig_tf_clusters):

            # Define figure name
            fname = f'GA_{region.name}_{l_freq}_{h_freq}'
            title = fname
            if active_times:
                fname += f"_{active_times[0]}_{active_times[1]}"

            # Plot
            fig, ax = plt.subplots(figsize=(10, 7))
            tfr.plot(axes=ax, mask=plot_mask, mask_style='contour', show=display_figs, title=title)[0]
            ax.vlines(x=0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyles='--', colors='gray')
            fig.suptitle(fname)

            if save_fig:
                save.fig(fig=fig, path=fig_path_diff + f'{cluster}/', fname=fname)

            # plot brain regions
            brain.add_label(region, borders=False)

        # Save brain plot
        if save_fig:
            brain.save_image(filename=fig_path_diff + f'{cluster}/' + 'brain_regions.png')
            brain.save_image(filename=fig_path_diff + f'{cluster}/svg/' + 'brain_regions.pdf')

        average_tf_and_significance_heatmap(generic_tfr=sig_cluster_regions[0], sig_tfr=sig_cluster_tfr, sig_mask=sig_tf_clusters, sig_regions=sig_cluster_regions,
                                            sig_chs_percent=sig_chs_percent, hist_data=None, active_times=active_times, l_freq=l_freq, h_freq=h_freq,
                                            display_figs=display_figs, save_fig=save_fig, fig_path=fig_path_diff + f'{cluster}/')


def quadrant_regions(quadrants_regions, sig_regions, sig_tfr, clusters_masks, sig_chs_percent, p_threshold, l_freq, h_freq, active_times, hist_data, subjects_dir, display_figs,
                     save_fig, fig_path_diff):

    for quadrant in quadrants_regions.keys():

        # Get clusters regions
        quadrant_regions = quadrants_regions[quadrant]
        quadrant_regions_idx = [i for i in range(len(sig_regions)) if sig_regions[i] in quadrant_regions]

        if len(quadrant_regions_idx):

            # Close mne 3d figures
            try:
                mne.viz.close_all_3d_figures()
            except:
                pass

            # Define brain for plotting
            Brain = mne.viz.get_brain_class()
            brain = Brain("fsaverage", hemi="split", surf="pial", views=['lat', 'med', 'dorsal'], subjects_dir=subjects_dir, size=(1080, 720))

            # Get cluster's regions data
            sig_quadrant_regions = [sig_regions[idx] for idx in quadrant_regions_idx]
            sig_quadrant_tfr = [sig_tfr[idx] for idx in quadrant_regions_idx]
            sig_mask_quadrant = [clusters_masks[idx] for idx in quadrant_regions_idx]

            # Iterate over quadrant's regions
            for region, tfr, plot_mask in zip(sig_quadrant_regions, sig_quadrant_tfr, sig_mask_quadrant):

                # Define figure name
                fname = f'GA_{region.name}_{l_freq}_{h_freq}'
                title = fname
                if active_times:
                    fname += f"_{active_times[0]}_{active_times[1]}"

                # Plot
                fig = plot_general.source_tf(tf=tfr, clusters_mask_plot=plot_mask, p_threshold=p_threshold, hist_data=hist_data, display_figs=display_figs,
                                             save_fig=save_fig, fig_path=fig_path_diff + f'{quadrant}/', fname=fname, title=title)

                # Close figure
                if not display_figs:
                   plt.close(fig)

                # plot brain regions
                brain.add_label(region, borders=False)

            if save_fig:
                # Save brain plot
                brain.save_image(filename=fig_path_diff + f'{quadrant}/' + 'brain_regions.png')
                brain.save_image(filename=fig_path_diff + f'{quadrant}/svg/' + 'brain_regions.pdf')

            # Take quadrant average and plot
            plot_general.average_tf_and_significance_heatmap(generic_tfr=sig_quadrant_tfr[0], sig_tfr=sig_quadrant_tfr, sig_mask=sig_mask_quadrant, sig_regions=sig_quadrant_regions,
                                                             sig_chs_percent=sig_chs_percent, p_threshold=p_threshold, l_freq=l_freq, h_freq=h_freq, hist_data=hist_data, active_times=active_times,
                                                             display_figs=display_figs, save_fig=save_fig, fig_path=fig_path_diff + f'{quadrant}/')


def compute_regional_connectivity_averages(con, labels, region_labels, parcellation='aparc.a2009s',
                                           n_connections=None, threshold_absolute=True):
    """
    Compute average connectivity for each brain region by averaging rows of the connectivity matrix.

    Parameters:
    -----------
    con : numpy.ndarray
        Connectivity matrix (n_labels x n_labels)
    labels : list
        List of label objects or label names
    region_labels : dict
        Dictionary mapping parcellation schemes to region definitions
    parcellation : str
        Parcellation scheme to use (default: 'aparc.a2009s')
    n_connections : int or None
        Number of strongest connections to keep for averaging. If None, use all connections.
    threshold_absolute : bool
        If True, threshold based on absolute values. If False, threshold based on positive values only.

    Returns:
    --------
    regional_averages : dict
        Dictionary with region names as keys and average connectivity arrays as values
    """

    # Get label names
    if hasattr(labels[0], 'name'):
        label_names = [label.name for label in labels]
    else:
        label_names = labels

    # Function to classify labels into brain regions
    def classify_region(label_name):
        if parcellation in region_labels:
            label_lower = label_name.lower()
            for region, region_terms in region_labels[parcellation].items():
                if any(term in label_lower for term in region_terms):
                    return region
        return 'unknown'

    # Group labels by region
    region_indices = {}
    for i, label_name in enumerate(label_names):
        region = classify_region(label_name)
        if region not in region_indices:
            region_indices[region] = []
        region_indices[region].append(i)

    # Apply threshold to connectivity matrix if specified
    if n_connections is not None:
        con_thresholded = con.copy()

        # Get threshold value based on strongest connections
        if threshold_absolute:
            # Use absolute values for thresholding
            threshold_value = np.sort(np.abs(con.flatten()))[-n_connections]
            mask = np.abs(con) < threshold_value
        else:
            # Use only positive values for thresholding
            positive_values = con[con > 0]
            if len(positive_values) >= n_connections:
                threshold_value = np.sort(positive_values)[-n_connections]
                mask = con < threshold_value
            else:
                # If fewer positive connections than requested, keep all positive
                mask = con <= 0

        # Zero out connections below threshold
        con_thresholded[mask] = 0
    else:
        con_thresholded = con

    # Compute regional averages
    regional_averages = {}
    for region, indices in region_indices.items():
        if len(indices) > 0:
            # Average the rows corresponding to this region
            regional_averages[region] = np.mean(con_thresholded[indices, :])

    return regional_averages
