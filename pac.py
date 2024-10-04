from tensorpac import Pac
from tensorpac import EventRelatedPac
import functions_general
import mne
import mne.beamformer as beamformer
import save
from paths import paths
import load
import setup
import numpy as np
import plot_general
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import itertools
from tqdm import tqdm

def add_task_lines(y_text, fontsize=10, color='white', ax=None):
    lines_times = [0, 2, 3]
    text_times = [-.32, 1, 2.5, 3.75]
    x_conditions = ['Cross 1', 'MS', 'Cross 2', 'VS']
    if ax is None: ax = plt.gca()  # noqa
    plt.sca(ax)
    for line_time in lines_times:
        plt.axvline(line_time, lw=2, color=color)
    for x_t, t_t in zip(text_times, x_conditions):
        plt.text(x_t, y_text, t_t, color=color, fontsize=fontsize, ha='center',
                 va='center', fontweight='bold')

# Load experiment info
exp_info = setup.exp_info()

#--------- Define Parameters ---------#

# Save
save_fig = True
save_data = True
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

#----- Parameters -----#
# Trial selection
trial_params = {'epoch_id': 'ms',  # use'+' to mix conditions (red+blue)
                'corrans': None,
                'tgtpres': None,
                'mss': 1,
                'reject': None,  # None to use default {'mag': 5e-12} / False for no rejection / 'subject' to use subjects predetermined rejection value
                'evtdur': None,
                'trialdur': None}

meg_params = {'chs_id': 'occipital',
              'filter_sensors': True,
              'filter_method': 'iir',
              'data_type': 'ICA'
              }

# Define PAC parameters
l_freq_amp, h_freq_amp = 20, 100
width_amp = 30
step_amp = 2

l_freq_pha, h_freq_pha = 4, 8
width_pha = 2
step_pha = .2

#--------- Setup ---------#

# Windows durations
cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()

# Get time limits based on epoch id
time_map = {'vs': dict(tmin=0, tmax=2, plot_xlim=(0, 2)),
            'ms': dict(tmin=-0.75, tmax=7, plot_xlim=(0, 7))}
trial_params['tmin'], trial_params['tmax'], plot_xlim = functions_general.get_time_lims(epoch_id=trial_params['epoch_id'], mss=trial_params['mss'], map=time_map)

# Get baseline duration for epoch_id
trial_params['baseline'], trial_params['plot_baseline'] = functions_general.get_baseline_duration(epoch_id=trial_params['epoch_id'], mss=trial_params['mss'],
                                                                                              tmin=trial_params['tmin'], tmax=trial_params['tmax'],
                                                                                              cross1_dur=cross1_dur, mss_duration=mss_duration,
                                                                                              cross2_dur=cross2_dur)

if 'vs' in trial_params['epoch_id'] and 'fix' not in trial_params['epoch_id'] and 'sac' not in trial_params['epoch_id']:
    trial_dur = vs_dur[trial_params['mss']]  # Edit this to determine the minimum visual search duration for the trial selection (this will only affect vs epoching)
else:
    trial_dur = None

# Define PAC computer
p_obj = Pac(idpac=(6, 0, 0), f_pha=(l_freq_pha, h_freq_pha, width_pha, step_pha), f_amp=(l_freq_amp, h_freq_amp, width_amp, step_amp))

# Define ERPAC computer
rp_obj = EventRelatedPac(f_pha=[l_freq_pha, h_freq_pha], f_amp=(l_freq_amp, h_freq_amp, width_amp, step_amp))


# Save ids
save_id = f"{trial_params['epoch_id']}_mss{trial_params['mss']}_corrans{trial_params['corrans']}_tgtpres{trial_params['tgtpres']}_trialdur{trial_dur}_evtdur{trial_params['evtdur']}"
# Redefine save id
if 'rel_sac' in trial_params.keys() and trial_params['rel_sac'] != None:
    save_id = trial_params['rel_sac'] + '_' + save_id

epochs_save_path = paths().save_path() + f"Epochs_{meg_params['data_type']}/Band_None/{save_id}_{trial_params['tmin']}_{trial_params['tmax']}_bline{trial_params['baseline']}/"

# Save figures paths
fig_path = paths().plots_path() + f"PAC_{meg_params['data_type']}/" + save_id + f"_{l_freq_pha}-{h_freq_pha}_{l_freq_amp}-{h_freq_amp}/{meg_params['chs_id']}"

# Save pac across subjects
pac_subjects_cross1 = []
pac_subjects_ms = []
pac_subjects_cross2 = []
pac_subjects_vs = []
erpac_subjects = []

# Iterate over subjects
for subject_code in exp_info.subjects_ids:

    # Define save path and file name for loading and saving epoched, evoked, and GA data
    if meg_params['data_type'] == 'ICA':
        # Load subject object
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
    else:
        # Load subject object
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

    # Data filenames
    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'

    # Load epoched data
    epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)

    # Pick channels
    picks = functions_general.pick_chs(chs_id=meg_params['chs_id'], info=epochs.info)
    epochs.pick(picks)

    pac_electrodes_cross1 = []
    pac_electrodes_ms = []
    pac_electrodes_cross2 = []
    pac_electrodes_vs = []
    # Compute PAC over electrodes
    print('Computing PAC')
    for e in tqdm(range(len(epochs.ch_names))):

        epochs_data = epochs.get_data()[:, e, :].squeeze()

        # Extract all phases and amplitudes
        epochs_pha = p_obj.filter(epochs.info['sfreq'], epochs_data, ftype='phase')
        epochs_amp = p_obj.filter(epochs.info['sfreq'], epochs_data, ftype='amplitude')

        # Change_times
        screen_change_times = [epochs.times[0] + cross1_dur,
                               epochs.times[0] + cross1_dur + mss_duration[trial_params['mss']],
                               epochs.times[0] + cross1_dur + + mss_duration[trial_params['mss']] + cross2_dur]

        screen_onsets, _ = functions_general.find_nearest(epochs.times, screen_change_times)

        # Extract screens data
        time_cross1 = slice(0, screen_onsets[0])
        time_ms = slice(screen_onsets[0], screen_onsets[1])
        time_cross2 = slice(screen_onsets[1], screen_onsets[2])
        time_vs = slice(screen_onsets[2], len(epochs.times))

        epochs_pha_cross1, epochs_amp_cross1 = epochs_pha[..., time_cross1], epochs_amp[..., time_cross1]
        epochs_pha_ms, epochs_amp_ms= epochs_pha[..., time_ms], epochs_amp[..., time_ms]
        epochs_pha_cross2, epochs_amp_cross2 = epochs_pha[..., time_cross2], epochs_amp[..., time_cross2]
        epochs_pha_vs, epochs_amp_vs = epochs_pha[..., time_vs], epochs_amp[..., time_vs]

        # Compute PAC inside rest, planning, and execution
        pac_cross1 = p_obj.fit(epochs_pha_cross1, epochs_amp_cross1, verbose=False).mean(-1)
        pac_ms = p_obj.fit(epochs_pha_ms, epochs_amp_ms, verbose=False).mean(-1)
        pac_cross2 = p_obj.fit(epochs_pha_cross2, epochs_amp_cross2, verbose=False).mean(-1)
        pac_vs = p_obj.fit(epochs_pha_vs, epochs_amp_vs, verbose=False).mean(-1)

        # Append
        pac_electrodes_cross1.append(pac_cross1)
        pac_electrodes_ms.append(pac_ms)
        pac_electrodes_cross2.append(pac_cross2)
        pac_electrodes_vs.append(pac_vs)

    # Average electrodes
    pac_subject_cross1 = np.mean(np.array(pac_electrodes_cross1), axis=0).squeeze()
    pac_subject_ms = np.mean(np.array(pac_electrodes_ms), axis=0).squeeze()
    pac_subject_cross2 = np.mean(np.array(pac_electrodes_cross2), axis=0).squeeze()
    pac_subject_vs = np.mean(np.array(pac_electrodes_vs), axis=0).squeeze()

    # Plot
    fname = f'{subject.subject_id}_pac_cross1'
    fig = plt.figure()
    p_obj.comodulogram(pac_subject_cross1)
    if save_fig:
        save.fig(fig=fig, path=fig_path, fname=fname)

    fname = f'{subject.subject_id}_pac_ms'
    fig = plt.figure()
    p_obj.comodulogram(pac_subject_ms)
    if save_fig:
        save.fig(fig=fig, path=fig_path, fname=fname)

    fname = f'{subject.subject_id}_pac_cross2'
    fig = plt.figure()
    p_obj.comodulogram(pac_subject_cross2)
    if save_fig:
        save.fig(fig=fig, path=fig_path, fname=fname)

    fname = f'{subject.subject_id}_pac_vs'
    fig = plt.figure()
    p_obj.comodulogram(pac_subject_vs)
    if save_fig:
        save.fig(fig=fig, path=fig_path, fname=fname)


    erpac_electrodes = []
    # Compute ERPAC over electrodes
    print('Computing ERPAC')
    for e in tqdm(range(len(epochs.ch_names))):
        epochs_data = epochs.get_data()[:, e, :].squeeze()

        # Compute ERPac
        erpac = rp_obj.filterfit(epochs.info['sfreq'], epochs_data, method='gc', smooth=100, verbose=False)

        # Append
        erpac_electrodes.append(erpac)

    # Avergae electrodes
    erpac_subject = np.mean(np.array(erpac_electrodes), axis=0).squeeze()

    # Plot
    fname = f'{subject.subject_id}_erpac'
    fig = plt.figure()
    rp_obj.pacplot(erpac_subject, epochs.times, rp_obj.yvec, xlabel='Time', ylabel='Amplitude frequency (Hz)',
                   title=f'Event-Related PAC occurring for {(l_freq_pha, h_freq_pha)} phase', fz_labels=15, fz_title=18)
    add_task_lines(y_text=(h_freq_amp - l_freq_amp) * 95/100)
    if save_fig:
        save.fig(fig=fig, path=fig_path, fname=fname)

    # Append to subjects data
    pac_subjects_cross1.append(pac_subject_cross1)
    pac_subjects_ms.append(pac_subject_ms)
    pac_subjects_cross2.append(pac_subject_cross2)
    pac_subjects_vs.append(pac_subject_vs)

    erpac_subjects.append(erpac_subject)


# Convert to array
pac_subjects_cross1 = np.array(pac_subjects_cross1)
pac_subjects_ms = np.array(pac_subjects_ms)
pac_subjects_cross2 = np.array(pac_subjects_cross2)
pac_subjects_vs = np.array(pac_subjects_vs)

erpac_subjects = np.array(erpac_subjects)

pac_ga_cross1 = np.mean(pac_subjects_cross1, axis=0)
pac_ga_ms = np.mean(pac_subjects_ms, axis=0)
pac_ga_cross2 = np.mean(pac_subjects_cross2, axis=0)
pac_ga_vs = np.mean(pac_subjects_vs, axis=0)

erpac_ga = np.mean(erpac_subjects, axis=0)

fname = 'GA_comodulogram_cross1'
fig = plt.figure()
p_obj.comodulogram(pac_ga_cross1)
if save_fig:
    save.fig(fig=fig, path=fig_path, fname=fname)

fname = 'GA_comodulogram_ms'
fig = plt.figure()
p_obj.comodulogram(pac_ga_ms)
if save_fig:
    save.fig(fig=fig, path=fig_path, fname=fname)

fname = 'GA_comodulogram_cross2'
fig = plt.figure()
p_obj.comodulogram(pac_ga_cross2)
if save_fig:
    save.fig(fig=fig, path=fig_path, fname=fname)

fname = 'GA_comodulogram_vs'
fig = plt.figure()
p_obj.comodulogram(pac_ga_vs)
if save_fig:
    save.fig(fig=fig, path=fig_path, fname=fname)

fname = 'GA_erpac'
fig = plt.figure()
rp_obj.pacplot(erpac_ga, epochs.times, rp_obj.yvec, xlabel='Time', ylabel='Amplitude frequency (Hz)',
               title=f'GA Event-Related PAC occurring for {(l_freq_pha, h_freq_pha)} phase', fz_labels=15, fz_title=18)
add_task_lines(y_text=(h_freq_amp - l_freq_amp) * 95/100)
if save_fig:
    save.fig(fig=fig, path=fig_path, fname=fname)