import os
import matplotlib.pyplot as plt
import mne
import functions_general
import functions_analysis
import load
import plot_general
import setup
import save
from paths import paths
from mne.preprocessing import ICA


#----- Paths -----#
save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()


#----- Save data and display figures -----#
save_data = False
save_fig = False
display_figs = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

plot_epochs = True
plot_gaze = True


#----- Parameters -----#
# ICA vs raw data
use_ica_data = False
# Frequency band
band_id = None
# Id
epoch_id = 'l_sac'
# Duration
dur = None  # seconds
# Plot channels
chs_id = 'mag'


# Screen
screen = functions_general.get_screen(epoch_id=epoch_id)
# MSS
mss = functions_general.get_mss(epoch_id=epoch_id)
# Item
tgt = functions_general.get_item(epoch_id=epoch_id)
# Saccades direction
dir = functions_general.get_dir(epoch_id=epoch_id)
# Get time windows from epoch_id name
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id)
# Specific run path for saving data and plots
run_path = f'/Band_{band_id}/{epoch_id}_{tmin}_{tmax}/'


#----- Run -----#
evokeds = []
for subject_code in exp_info.subjects_ids[4:]:

    # Define save path and file name for loading and saving epoched, evoked, and GA data

    # Load subject object
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    # Save data paths
    epochs_save_path = save_path + f'Epochs_RAW/' + run_path
    evoked_save_path = save_path + f'Evoked_RAW/' + run_path
    # Save figures paths
    epochs_fig_path = plot_path + f'Epochs_RAW/' + run_path
    evoked_fig_path = plot_path + f'Evoked_RAW/' + run_path

    # Data filenames
    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
    evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'

    try:
        # Load epoched data
        epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
        # Pick MEG channels to plot
        picks = functions_general.pick_chs(chs_id=chs_id, info=epochs.info)
    except:
        # Compute
        meg_data = subject.load_preproc_meg()

        # Pick MEG channels to plot
        picks = functions_general.pick_chs(chs_id=chs_id, info=meg_data.info)

        # Exclude bad channels
        bads = subject.bad_channels
        meg_data.info['bads'].extend(bads)

        metadata, events, events_id, metadata_sup = functions_analysis.define_events(subject=subject, epoch_id=epoch_id,
                                                                                     screen=screen, mss=mss, dur=dur,
                                                                                     tgt=tgt, dir=dir, meg_data=meg_data)

        # Reject based on channel amplitude
        reject = dict(mag=subject.config.general.reject_amp)

        # Epoch data
        epochs = mne.Epochs(raw=meg_data, events=events, event_id=events_id, tmin=tmin, tmax=tmax, reject=reject,
                            event_repeated='drop', metadata=metadata, preload=True)
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

    if plot_epochs:
        # Parameters for plotting
        overlay = epochs.metadata['duration']
        if overlay is not None:
            order = overlay.argsort()  # Sorting from longer to shorter
        else:
            order = None
        sigma = 5
        combine = 'std'
        group_by = {}

        # Figure file name
        fname = 'Epochs_' + subject.subject_id + f'_{chs_id}_{combine}'

        # Plot epochs
        plot_general.epochs(subject=subject, epochs=epochs, picks=picks, order=order, overlay=overlay, combine=combine, sigma=sigma,
                            group_by=group_by, display_figs=display_figs, save_fig=save_fig, fig_path=epochs_fig_path, fname=fname)

    # Evoked for plotting before and after
    evoked = epochs.average(picks=['mag'])

    # Fit ICA to epochs and plot overlay
    ica_components = 64
    lfreq = 1
    hfreq = 40
    epochs.filter(lfreq, hfreq)

    # Define ICA
    ica = ICA(method='infomax', random_state=97, n_components=ica_components)

    # Apply ICA
    ica.fit(epochs)

    # Plot components and signals
    ica.plot_sources(epochs, title='ICA')
    ica.plot_components()

    # Select bad components
    ex_components = [1,2,9,17,18,33,39,43,61,62,63]

    # plot before and after
    ica.plot_overlay(evoked, exclude=ex_components)

    # Save components figures
    # Create directory
    fig_path = plot_path + f'ICA_on_Epochs/{subject.subject_id}/'
    os.makedirs(fig_path, exist_ok=True)

    # Plot properties of excluded components
    all_comps = [i for i in range(ica_components)]
    ica.plot_properties(epochs, picks=all_comps, psd_args=dict(fmax=hfreq), show=False)

    # Get figures
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for i, fig in enumerate(figs):
        save.fig(fig=fig, path=fig_path, fname=f'figure_{i}')
    plt.close('all')

    # Plot properties of excluded components
    ica.plot_properties(epochs, picks=ex_components, psd_args=dict(fmax=hfreq), show=False)

    # Get figures
    figs = [plt.figure(n) for n in plt.get_fignums()]
    # Redefine save path
    fig_path += 'Excluded/'
    for i, fig in enumerate(figs):
        save.fig(fig=fig, path=fig_path, fname=f'figure_{i}')

