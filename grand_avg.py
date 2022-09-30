import load
import os
import mne

from paths import paths

save_path = paths().save_path()
reject = dict(mag=4e-12)
l_freq = 0.5
h_freq = 100
epoch_id = 'fix_vs'

evokeds = []

for subject in [0, 1, 2, 3, 4, 5]:

    subject = load.subject(subject)
    evoked_save_path = save_path + f'Evoked/{epoch_id}_lfreq{l_freq}_hfreq{h_freq}/' + subject.subject_id + '/'
    evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'
    evoked = mne.read_evokeds(evoked_data_fname)
    evokeds.append(evoked)

    # Estimate sources, save sources and average sources

# Compute grand average
grand_avg = mne.grand_average(evokeds, interpolate_bads=False)
