import pandas as pd
import functions_general
import plot_preproc
from paths import paths
import setup
import load
import save
import matplotlib.pyplot as plt
import numpy as np

save_fig = False
save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()

all_fixations = pd.DataFrame()
all_saccades = pd.DataFrame()
all_bh_data = pd.DataFrame()
all_rt = pd.DataFrame()
all_mss = pd.DataFrame()
all_corr_ans = pd.DataFrame()

# Saccades, tgt fixations and response times relative to end time
all_saccades_end = pd.DataFrame()
all_fixations_target = pd.DataFrame()
all_response_times_end = pd.DataFrame()

all_acc = {1: [], 2: [], 4: []}
all_response_times = {1: [], 2: [], 4: []}

for subject_code in exp_info.subjects_ids:

    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    print(f'\nSubject {subject.subject_id}')

    print(f'Total fixations: {subject.len_all_fix}')
    print(f'Total saccades: {subject.len_all_sac}')

    print(f'Drop fixations: {subject.len_fix_drop}')
    print(f'Drop saccades: {subject.len_sac_drop}')

    print(f'Final fixations: {len(subject.fixations)}')
    print(f'Final saccades: {len(subject.saccades)}')

    fixations = subject.fixations
    saccades = subject.saccades
    bh_data = subject.bh_data
    rt = pd.DataFrame(subject.rt)
    corr_ans = pd.DataFrame(subject.corr_ans)
    trial_mss = pd.DataFrame(subject.bh_data['Nstim'])

    cross1_fixations = fixations.loc[fixations['screen'] == 'cross1']
    ms_fixations = fixations.loc[fixations['screen'] == 'ms']
    cross2_fixations = fixations.loc[fixations['screen'] == 'cross2']
    vs_fixations = fixations.loc[fixations['screen'] == 'vs']

    cross1_saccades = saccades.loc[saccades['screen'] == 'cross1']
    ms_saccades = saccades.loc[saccades['screen'] == 'ms']
    cross2_saccades = saccades.loc[saccades['screen'] == 'cross2']
    vs_saccades = saccades.loc[saccades['screen'] == 'vs']

    # Saccades times distributions
    saccades_vs = saccades.loc[saccades['screen'] == 'vs']
    saccades_end_subj = []
    for index, saccade in saccades_vs.iterrows():
        trial_idx = saccade.trial - 1
        vs_dur = subject.vsend[trial_idx] - subject.vs[trial_idx]
        saccades_end_subj.append(saccade.delay - float(vs_dur))

    # tgt_fix times distributions
    tgt_fixations = fixations.loc[fixations['fix_target'] == 1]
    tgt_fixations_subj = []
    for index, fixation in tgt_fixations.iterrows():
        trial_idx = fixation.trial - 1
        vs_dur = subject.vsend[trial_idx] - subject.vs[trial_idx]
        tgt_fixations_subj.append(fixation.delay - float(vs_dur))

    # Response times distributions reslative to trial end
    response_times_end = []
    for trial_idx, response_time in rt.iterrows():
        vs_dur = subject.vsend[trial_idx] - subject.vs[trial_idx]
        response_times_end.append(response_time.values[0] - float(vs_dur))

    all_saccades_end = pd.concat([all_saccades_end, pd.Series(saccades_end_subj)])
    all_fixations_target = pd.concat([all_fixations_target, pd.Series(tgt_fixations_subj)])
    all_response_times_end = pd.concat([all_response_times_end, pd.Series(response_times_end)])

    all_fixations = pd.concat([all_fixations, fixations])
    all_saccades = pd.concat([all_saccades, saccades])
    all_bh_data = pd.concat([all_bh_data, bh_data])
    all_rt = pd.concat([all_rt, rt])
    all_mss = pd.concat([all_mss, trial_mss])
    all_corr_ans = pd.concat([all_corr_ans, corr_ans])

    # Plot performance to extract values and save for GA
    corr1_mean, corr2_mean, corr4_mean, rt1_mean, rt2_mean, rt4_mean = plot_preproc.performance(subject=subject, display=False, save_fig=False)

    all_acc[1].append(corr1_mean)
    all_acc[2].append(corr2_mean)
    all_acc[4].append(corr4_mean)

    all_response_times[1].append(rt1_mean)
    all_response_times[2].append(rt2_mean)
    all_response_times[4].append(rt4_mean)

# Define all subjects class instance
subjects = setup.all_subjects(all_fixations, all_saccades, all_bh_data, all_rt.values, all_corr_ans.values, all_mss)

plot_preproc.first_fixation_delay(subject=subjects)
plot_preproc.pupil_size_increase(subject=subjects)


plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(2, 3, figsize=(12, 7))

plot_preproc.all_subj_performance(axs, all_acc, all_response_times)
plot_preproc.fixation_duration(subject=subjects, ax=axs[0, 1])
plot_preproc.saccades_amplitude(subject=subjects, ax=axs[0, 2])
plot_preproc.saccades_dir_hist(subject=subjects, fig=fig, axs=axs, ax_idx=4)
plot_preproc.sac_main_seq(subject=subjects, ax=axs[1, 2])

fig.tight_layout()

if save_fig:
    save_path = paths().plots_path() + 'Preprocessing/' + subjects.subject_id + '/'
    fname = f'{subjects.subject_id} Multipanel'
    save.fig(fig=fig, path=save_path, fname=fname)


## Fixation and saccade rates

import numpy as np
import load
import plot_general
import functions_general
import setup
import save
import matplotlib.pyplot as plt
from paths import paths

save_fig = True
display_figs = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()
cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()

# All subjects fixation and saccades rate
all_fixations_rate = {}
all_saccades_rate = {}

for mss in [1, 2, 4]:
    all_fixations_rate[mss] = []
    all_saccades_rate[mss] = []

    for subject_code in exp_info.subjects_ids:

        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
        print(f'\nSubject {subject.subject_id}')

        print(f'Total fixations: {subject.len_all_fix}')
        print(f'Total saccades: {subject.len_all_sac}')

        print(f'Drop fixations: {subject.len_fix_drop}')
        print(f'Drop saccades: {subject.len_sac_drop}')

        print(f'Final fixations: {len(subject.fixations)}')
        print(f'Final saccades: {len(subject.saccades)}')

        fixations = subject.fixations
        saccades = subject.saccades
        bh_data = subject.bh_data

        mss_trials_idx = bh_data.loc[bh_data['Nstim'] == mss].index

        fixations_mss = fixations.loc[fixations['mss'] == mss]
        saccades_mss = saccades.loc[saccades['mss'] == mss]

        # Consider delay since trial onset
        fixations_mss.loc[(fixations_mss['mss'] == mss) & (fixations_mss['screen'] == 'ms'), 'delay'] += cross1_dur
        saccades_mss.loc[(saccades_mss['mss'] == mss) & (saccades_mss['screen'] == 'ms'), 'delay'] += cross1_dur

        fixations_mss.loc[(fixations_mss['mss'] == mss) & (fixations_mss['screen'] == 'cross2'), 'delay'] += cross1_dur + mss_duration[mss]
        saccades_mss.loc[(saccades_mss['mss'] == mss) & (saccades_mss['screen'] == 'cross2'), 'delay'] += cross1_dur + mss_duration[mss]

        fixations_mss.loc[(fixations_mss['mss'] == mss) & (fixations_mss['screen'] == 'vs'), 'delay'] += cross1_dur + mss_duration[mss] + cross2_dur
        saccades_mss.loc[(saccades_mss['mss'] == mss) & (saccades_mss['screen'] == 'vs'), 'delay'] += cross1_dur + mss_duration[mss] + cross2_dur

        # Sliding window
        time_window = 0.3
        time_step = 0.01
        total_trial_length = cross1_dur + mss_duration[mss] + cross2_dur + vs_dur[4][0]

        fixations_num = []
        saccades_num = []
        trials_num = []

        t = 0
        while t <= total_trial_length - time_window:

            fixations_num.append(len(fixations_mss.loc[(fixations_mss['delay'] >= t) & (fixations_mss['delay'] <= t + time_window)]))
            saccades_num.append(len(saccades_mss.loc[(saccades_mss['delay'] >= t) & (saccades_mss['delay'] <= t + time_window)]))

            trials_dur = cross1_dur + mss_duration[mss] + cross2_dur + (subject.vsend - subject.vs)
            trials_num.append(sum(trials_dur[mss_trials_idx] >= t))

            t += time_step

        if len(trials_num) == len(fixations_num) == len(saccades_num):
            fixations_rate = [fixations_num[i] / trials_num[i] / time_window for i in range(len(trials_num))]
            saccades_rate = [saccades_num[i] / trials_num[i] / time_window for i in range(len(trials_num))]

        all_fixations_rate[mss].append(fixations_rate)
        all_saccades_rate[mss].append(saccades_rate)

    fixations_rate_mean = np.nanmean(all_fixations_rate[mss], axis=0)
    fixations_rate_std = np.nanstd(all_fixations_rate[mss], axis=0)

    saccades_rate_mean = np.nanmean(all_saccades_rate[mss], axis=0)
    saccades_rate_std = np.nanstd(all_saccades_rate[mss], axis=0)

    # Plot Fixations
    fig, axes_topo, ax_tf, ax_cbar, ax_tfr_cbar = plot_general.fig_tf_times(time_len=total_trial_length, ax_len_div=24)
    title = f'Mean fixation rate (MSS= {mss} - Window= {int(time_window * 1000)} ms)'
    fig.suptitle(title)
    ax_tf.plot(np.linspace(0, total_trial_length, len(fixations_rate_mean)), fixations_rate_mean)
    ax_tf.fill_between(x=np.linspace(0, total_trial_length, len(fixations_rate_mean)), y1=fixations_rate_mean - fixations_rate_std, y2=fixations_rate_mean + fixations_rate_std, alpha=0.5)

    # Plot vlines
    ymin = ax_tf.get_ylim()[0]
    ymax = ax_tf.get_ylim()[1]
    for t in [cross1_dur, cross1_dur + mss_duration[mss], cross1_dur + mss_duration[mss] + cross2_dur]:
        ax_tf.vlines(x=t, ymin=ymin, ymax=ymax, linestyles='--', colors='gray')

    # Labels
    ax_tf.set_xlabel('Time (s)')
    ax_tf.set_ylabel('Saccade rate (saccades/s)')

    if save_fig:
        fig_path = paths().plots_path() + 'Preprocessing/All_Subjects/'
        save.fig(fig=fig, path=fig_path, fname=title)

    # Plot Saccades
    fig, axes_topo, ax_tf, ax_cbar, ax_tfr_cbar = plot_general.fig_tf_times(time_len=total_trial_length, ax_len_div=24)
    title = f'Mean saccades rate (MSS= {mss} - Window= {int(time_window * 1000)} ms)'
    fig.suptitle(title)
    ax_tf.plot(np.linspace(0, total_trial_length, len(fixations_rate_mean)), fixations_rate_mean)
    ax_tf.fill_between(x=np.linspace(0, total_trial_length, len(fixations_rate_mean)), y1=fixations_rate_mean - fixations_rate_std,
                       y2=fixations_rate_mean + fixations_rate_std, alpha=0.5)

    # Plot vlines
    ymin = ax_tf.get_ylim()[0]
    ymax = ax_tf.get_ylim()[1]
    for t in [cross1_dur, cross1_dur + mss_duration[mss], cross1_dur + mss_duration[mss] + cross2_dur]:
        ax_tf.vlines(x=t, ymin=ymin, ymax=ymax, linestyles='--', colors='gray')

    # Labels
    ax_tf.set_xlabel('Time (s)')
    ax_tf.set_ylabel('Saccade rate (saccades/s)')

    if save_fig:
        fig_path = paths().plots_path() + 'Preprocessing/All_Subjects/'
        save.fig(fig=fig, path=fig_path, fname=title)


# All MSS plots
plot_edge = 0.15
x_drop_size = int(plot_edge / time_step)
fig1, axes_topo1, ax1, ax_cbar1, ax_tfr_cbar = plot_general.fig_tf_times(time_len=cross1_dur + mss_duration[4] + cross2_dur + vs_dur[4][0] - plot_edge*2, ax_len_div=24, fontsize=16, ticksize=16)
fig2, axes_topo2, ax2, ax_cbar2, ax_tfr_cbar = plot_general.fig_tf_times(time_len=cross1_dur + mss_duration[4] + cross2_dur + vs_dur[4][0] - plot_edge*2, ax_len_div=24, fontsize=16, ticksize=16)

for mss in [1, 2, 4]:

    fixations_rate_mean = np.nanmean(all_fixations_rate[mss], axis=0)
    fixations_rate_std = np.nanstd(all_fixations_rate[mss], axis=0)

    saccades_rate_mean = np.nanmean(all_saccades_rate[mss], axis=0)
    saccades_rate_std = np.nanstd(all_saccades_rate[mss], axis=0)

    plot_x_axis = np.linspace(- cross1_dur - mss_duration[mss] - cross2_dur + plot_edge, vs_dur[4][0] - plot_edge, len(fixations_rate_mean) - x_drop_size * 2)

    # Plot Fixations
    title1 = f'Mean fixation rate (Window= {int(time_window * 1000)} ms)'
    fig1.suptitle(title1)
    ax1.plot(plot_x_axis, fixations_rate_mean[x_drop_size:-x_drop_size], label=f'MSS: {mss}')
    ax1.fill_between(x=plot_x_axis, y1=fixations_rate_mean[x_drop_size:-x_drop_size] - fixations_rate_std[x_drop_size:-x_drop_size],
                     y2=fixations_rate_mean[x_drop_size:-x_drop_size] + fixations_rate_std[x_drop_size:-x_drop_size], alpha=0.5)
    ax1.legend()

    # Labels
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Fixations/s)')

    # Plot Saccades
    title2 = f'Mean saccades rate (Window= {int(time_window * 1000)} ms)'
    fig2.suptitle(title2)
    ax2.plot(plot_x_axis, fixations_rate_mean[x_drop_size:-x_drop_size], label=f'MSS: {mss}')
    ax2.fill_between(x=plot_x_axis, y1=fixations_rate_mean[x_drop_size:-x_drop_size] - fixations_rate_std[x_drop_size:-x_drop_size],
                     y2=fixations_rate_mean[x_drop_size:-x_drop_size] + fixations_rate_std[x_drop_size:-x_drop_size], alpha=0.5)
    ax2.legend()

    # Labels
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Saccades/s')

# Plot vlines
ymin1 = ax1.get_ylim()[0]
ymin2 = ax2.get_ylim()[0]
ymax1 = ax1.get_ylim()[1]
ymax2 = ax2.get_ylim()[1]

for t in [0, - cross2_dur, - mss_duration[1] - cross2_dur, - mss_duration[2] - cross2_dur, - mss_duration[4] - cross2_dur]:
    ax1.vlines(x=t, ymin=ymin1, ymax=ymax1, linestyles='--', colors='gray')
    ax2.vlines(x=t, ymin=ymin2, ymax=ymax2, linestyles='--', colors='gray')

# Remove blank space before and after
ax1.autoscale(tight=True)
ax2.autoscale(tight=True)

if save_fig:
    fig_path = paths().plots_path() + 'Preprocessing/All_Subjects/'
    save.fig(fig=fig1, path=fig_path, fname=title1)
    save.fig(fig=fig2, path=fig_path, fname=title2)


## Relative to end time plots
# Saccades
fig, ax = plt.subplots()
all_saccades_end.hist(range=[-2, 0], edgecolor='black', linewidth=1.2, density=True, stacked=True, ax=ax)
plt.title('VS saccades realtive to search end')
plt.xlabel('time [s]')
save_path = paths().plots_path()
fname = f'VS saccades trial end'
save.fig(fig=fig, path=save_path, fname=fname)

# Target fixations
fig, ax = plt.subplots()
all_fixations_target.hist(range=[-2, 0], edgecolor='black', linewidth=1.2, density=True, stacked=True, ax=ax)
plt.title('Target fixations realtive to search end')
plt.xlabel('time [s]')
save_path = paths().plots_path()
fname = f'Target fixations trial end'
save.fig(fig=fig, path=save_path, fname=fname)

# Response times
fig, ax = plt.subplots()
all_response_times_end.hist(range=[-0.03, 0], edgecolor='black', linewidth=1.2, density=True, stacked=True, ax=ax)
plt.title('Response time realtive to search end')
plt.xlabel('time [s]')
save_path = paths().plots_path()
fname = f'Response trime trial end'
save.fig(fig=fig, path=save_path, fname=fname)




## VS duration

vs_dur = subjects.rt
vs_dur = vs_dur.fillna(10)

fig, ax = plt.subplots()

ax.hist(vs_dur, bins=70, range=(0, 10), edgecolor='black', linewidth=1.2, density=False, stacked=True)
ax.set_title('VS duration')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Quantity')

threshold = 4
vs_dur_thr = vs_dur[vs_dur[0] > threshold]
ratio = len(vs_dur_thr)/len(vs_dur)*100
print(f'Percentage so samples kept using {threshold}s threshold: {round(ratio, 1)}%')


# VS duration by mss
inf_threshold = 3.5
sup_threshold = 9.8
mss = [1, 2, 4]
fig, axs = plt.subplots(nrows=len(mss), sharex=True)
fig.suptitle('VS duration')

for i, ax in enumerate(axs):
    vs_dur_mss = vs_dur[(subjects.mss == mss[i]).values]
    vs_dur_thr = vs_dur_mss[(vs_dur_mss[0] > inf_threshold) & (vs_dur_mss[0] < sup_threshold)]
    ratio = len(vs_dur_thr) / len(vs_dur_mss) * 100
    print(f'Percentage so samples kept using {inf_threshold}-{sup_threshold} s threshold for mss {mss[i]}: {round(ratio, 1)}%')

    ax.hist(vs_dur_mss, bins=70, range=(0, 10), edgecolor='black', linewidth=1.2, density=False, stacked=True)
    ax.set_ylabel(f'MSS {mss[i]}')
ax.set_xlabel('Time (s)')




## MS fixations heatmap
import numpy as np
import matplotlib.pyplot as plt

ms_fixations = all_fixations.loc[all_fixations['screen'] == 'ms']

# Plot heatmap
fixations_x = ms_fixations['mean_x']
fixations_y = ms_fixations['mean_y']

# items positions
it_x = np.array([-300, -150, 0, 150, 300])
it_y = np.array([-100, -50, 0, 50, 100])

# Transform to pixels with origin in top left
it_x_scaled = it_x + 1920 / 2
it_y_scaled = - it_y + 1080 / 2  # Take negative of items y position due to different coordinate system between psychopy (<0 lower half of the screen) and ET data (<0 upper half of the screen)

# Calculate the 2D histogram of fixations
fig, ax = plt.subplots()
h, xedged, yedges, im = plt.hist2d(fixations_x, -fixations_y, bins=(200, 200), cmap='hot', range=[[0, 1920], [-1024, 0]])
for x in it_x_scaled:
    for y in it_y_scaled:
        plt.plot(x, -y, 'o', color='C0')

ylabels = [str(item.get_text()).replace('−', '') for item in ax.get_yticklabels()]

ax.set_yticklabels(ylabels)
