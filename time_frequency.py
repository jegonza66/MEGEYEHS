import functions_general
import functions_analysis
import load
import mne

import plot_general
import matplotlib.pyplot as plt
import setup
from paths import paths

#----- Path -----#
save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()


#----- Save data and display figures -----#
save_data = True
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Select MEG channels -----#
'''
Select channels or set picks = 'mag')
'''
chs_id = 'mag'

#-----  Select frequency band -----#
band_id = 'Theta'

#----- Select events -----#
'''
Id Format
Saccades: f'{dir}_sac_{screen}_t{trial}_{n_sacs[-1]}'
Fixations: f'{prefix}_fix_{screen}_t{trial}_{n_fix}' prefix (tgt/it/none)only if vs screen
'''
evt_from_df = True
evt_from_annot = False

# MSS
mss = None
# Id
epoch_id = f'fix_ms'
# Duration
dur = 0.2  # seconds
# Direction
dir = None
# Screen
screen = epoch_id.split('_')[-1]
# Item
tgt = functions_general.get_item(epoch_id=epoch_id)

# Get time windows from epoch_id name
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id)

# Specific run path for saving data and plots
run_path = f'/{band_id}/{epoch_id}_{tmin}_{tmax}/'

evokeds = []
for subject_code in exp_info.subjects_ids:

    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    if band_id:
        meg_data = load.filtered_data(subject=subject, band_id=band_id)
    else:
        meg_data = subject.load_preproc_meg()

    # Exclude bad channels
    bads = subject.bad_channels
    meg_data.info['bads'].extend(bads)

    metadata, events, events_id = functions_analysis.define_events(subject=subject, epoch_id=epoch_id,
                                                                   evt_from_df=evt_from_df, evt_from_annot=evt_from_annot,
                                                                   screen=screen, mss=mss, dur=dur, tgt=tgt, dir=dir,
                                                                   meg_data=meg_data)

    # Reject based on channel amplitude
    reject = dict(mag=subject.config.general.reject_amp)

    # Epoch data
    epochs = mne.Epochs(raw=meg_data, events=events, event_id=events_id, tmin=tmin, tmax=tmax, reject=reject,
                        event_repeated='drop', metadata=metadata, preload=True)
    # Drop bad epochs
    epochs.drop_bad()

    # Parameters for plotting
    overlay = None
    if overlay:
        order = overlay.argsort()  # Sorting from longer to shorter
    else:
        order = None
    combine = 'mean'
    group_by = {}

    save_fig = True
    fig_path = plot_path + f'Epochs/' + run_path
    fname = 'Epochs_' + subject.subject_id + f'_{chs_id}_{combine}'