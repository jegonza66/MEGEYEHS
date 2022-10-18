import setup
import load
import save
import mne
import os
from paths import paths


save_path = paths().save_path()
plot_path = paths().plots_path()

l_freq = 0.5
h_freq = 100

epoch_ids = ['l_sac_emap']
plot_ind = False
save_ind = False

evokeds = []

for subject_code in range(9):

    exp_info = setup.exp_info()
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

    evoked_save_path = save_path + f'Evoked/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}/' + subject.subject_id + '/'
    evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'
    evoked = mne.read_evokeds(evoked_save_path + evoked_data_fname, condition=0)
    evokeds.append(evoked)

    if plot_ind:
        fig = evoked.plot(gfp=True, time_unit='s', spatial_colors=True, xlim=(-0.05, 0.1))
        if save_ind:
            fig_path = plot_path + f'Evoked/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}/'
            fname = subject.subject_id + '.png'
            save.fig(fig, fig_path, fname)

# Compute grand average
grand_avg = mne.grand_average(evokeds, interpolate_bads=False)
# Save grand average
ga_save_path = save_path + f'Evoked/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}/'
os.makedirs(ga_save_path, exist_ok=True)
grand_avg_data_fname = f'Grand_average_ave.fif'
grand_avg.save(ga_save_path + grand_avg_data_fname, overwrite=True)

# PLOT
fig = grand_avg.plot(gfp=True, spatial_colors=True, time_unit='s', xlim=(-0.05, 0.1), window_title=f'Grand average {"-".join(epoch_ids)}')
fig_path = plot_path + f'Evoked/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}/'
fname = 'Grand_average.png'
save.fig(fig, fig_path, fname)

# Saccades channels
sac_chs = ['MLF14-4123', 'MLF13-4123', 'MLF12-4123', 'MLF11-4123', 'MRF11-4123', 'MRF12-4123', 'MRF13-4123', 'MRF14-4123', 'MZF01-4123']
fig = grand_avg.plot(picks=sac_chs, gfp=True, spatial_colors=True, time_unit='s', xlim=(-0.05, 0.1), window_title=f'Grand average {"-".join(epoch_ids)}')
fname = 'Grand_average_ch_sel.png'
save.fig(fig, fig_path, fname)