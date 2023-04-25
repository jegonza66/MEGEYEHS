import matplotlib.pyplot as plt
import numpy as np
import mne
import functions_general
import functions_analysis
import load
import setup
from paths import paths
import pathlib
import os
import save

#----- Paths -----#
preproc_path = paths().preproc_path()
save_path = paths().save_path()
ica_path = paths().ica_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()


#----- Save data and display figures -----#
save_fig = True
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()


#----- Parameters -----#
# Subject
subject_code = exp_info.subjects_ids[5]
# Id
epoch_ids = ['l_sac', 'r_sac', 'u_sac', 'd_sac']
epoch_names = {'l_sac': 'Leftward', 'r_sac': 'Rightward', 'u_sac': 'Upward', 'd_sac': 'Downward'}


# Load subject object
subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
raw_data = subjec.load_preproc_meg_data()

# Load ica subject object
ica_subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
ica_data = load.ica_data(subject=subject)

# Load downsampled data for running ploch algorithm
sfreq = 200
lfreq = 1
hfreq = 40
downsampled_path = pathlib.Path(os.path.join(preproc_path, subject.subject_id, f'down-filt({sfreq}_{lfreq}_{hfreq})_meg.fif'))
downsampled_data = mne.io.read_raw_fif(downsampled_path, preload=False)

# Load ica
save_path_ica = ica_path + subject.subject_id + '/'
ica_fname = 'ICA.pkl'
ica = load.var(file_path=save_path_ica + ica_fname)
print('ICA object loaded')
# Get ICA components
print('Getting ICA sources')
ica_comps = ica.get_sources(raw_data)

# Get ocular components by Ploch's algorithm
ocular_components, sac_epochs_ds, fix_epochs_ds = \
    functions_analysis.ocular_components_ploch(subject=subject, meg_downsampled=downsampled_data,
                                               ica=ica, plot_distributions=False)
ocular_components_names = [f'ICA{component:03d}' for component in ocular_components]
ica_comps_subset = ica_comps.copy().pick(ocular_components_names)

# Compare signals RAW vs. ICA
# Select channels for comparison
plot_chs = ['MLT21', 'MLT31', 'MLT41', 'MLT51']
# Select time and samples
t_start = raw_data.annotations.onset[np.where(raw_data.annotations.description == 'vl_start')[0][4]]
t_end = raw_data.annotations.onset[np.where(raw_data.annotations.description == 'hs_start')[0][4]]
sample_start, _ = functions_general.find_nearest(array=raw_data.times, values=t_start)
sample_end, _ = functions_general.find_nearest(array=raw_data.times, values=t_end)

# RAW data
raw_subset = raw_data.copy().pick(plot_chs)
raw_subset_info = raw_subset.info
raw_subset_array = raw_subset.get_data(start=sample_start, stop=sample_end)
raw_evoked = mne.EvokedArray(raw_subset_array, raw_subset_info, tmin=t_start, baseline=(None, None))

# ICA data
ica_subset = ica_data.copy().pick(plot_chs)
ica_subset_info = ica_subset.info
ica_subset_array = ica_subset.get_data(start=sample_start, stop=sample_end)
ica_evoked = mne.EvokedArray(ica_subset_array, ica_subset_info, tmin=t_start, baseline=(None, None))

# Plot signal before and after
fig_sgn, axs = plt.subplots(nrows=2)
raw_evoked.plot(spatial_colors=True, axes=axs[0])
ica_evoked.plot(spatial_colors=True, axes=axs[1], ylim=dict(mag=[axs[0].get_ylim()[0], axs[0].get_ylim()[1]]))
axs[0].set_title('Raw data')
axs[1].set_title('ICA data')
axs[0].texts.pop(1)
axs[1].texts.pop(1)
fig_sgn.tight_layout()

# Save figure
fig_sgn_name = 'raw_vs_ica_signal'
save.fig(fig=fig_sgn, path=plot_path + 'ICA/', fname=fig_sgn_name)


# Plot components on epochs
plt.ion()
fig, axs = plt.subplots(nrows=len(ocular_components), ncols=len(epoch_ids)+1, figsize=(15,7))
topo_axes = axs[:, 0]

#----- Run -----#
for i, epoch_id in enumerate(epoch_ids):
    tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id)

    metadata, events, events_id, metadata_sup = functions_analysis.define_events(subject=subject, epoch_id=epoch_id,
                                                                                 screen=None, mss=None, dur=None,
                                                                                 tgt=None, dir=None, meg_data=ica_comps_subset)

    # Reject based on channel amplitude
    reject = dict(mag=subject.config.general.reject_amp)

    # Epoch data
    epochs = mne.Epochs(raw=ica_comps_subset, events=events, event_id=events_id, tmin=tmin, tmax=tmax, reject=None,
                        event_repeated='drop', metadata=metadata, preload=True, baseline=(None, None))
    # Drop bad epochs
    epochs.drop_bad()

    metadata_sup = metadata_sup.loc[(metadata_sup['id'].isin(epochs.metadata['event_name']))].reset_index(drop=True)
    epochs.metadata = metadata_sup

    evoked = epochs.average(picks=ocular_components_names)

    for j, component in enumerate(ocular_components_names):
        evoked.plot(picks=component, axes=axs[j, i+1], titles='')
        axs[j, i+1].texts.pop(1)
        if j==0:
            axs[j, i+1].set_title(epoch_names[epoch_id])
        else:
            axs[j, i + 1].set_title('')

ica.plot_components(picks=ocular_components, axes=topo_axes)
fig.tight_layout()
# Save figure
fig_name='ocular_components'
save.fig(fig=fig, path=plot_path + 'ICA/', fname=fig_name)

