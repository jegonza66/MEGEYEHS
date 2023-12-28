import pandas as pd
import plot_preproc
from paths import paths
import setup
import load
import save
import matplotlib.pyplot as plt

save_fig = False
save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

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
    mss = pd.DataFrame(subject.bh_data['Nstim'])

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
    all_mss = pd.concat([all_mss, mss])
    all_corr_ans = pd.concat([all_corr_ans, corr_ans])


# Relative to end time plots

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


# Define all subjects class instance
subjects = setup.all_subjects(all_fixations, all_saccades, all_bh_data, all_rt, all_corr_ans, all_mss)

plot_preproc.first_fixation_delay(subject=subjects)
plot_preproc.pupil_size_increase(subject=subjects)

plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplots(2, 2, figsize=(12, 7))

plot_preproc.fixation_duration(subject=subjects, ax=axs[0, 0])
plot_preproc.saccades_amplitude(subject=subjects, ax=axs[0, 1])
plot_preproc.saccades_dir_hist(subject=subjects, fig=fig, ax=axs[1, 0])
plot_preproc.sac_main_seq(subject=subjects, ax=axs[1, 1])

fig.tight_layout()

if save_fig:
    save_path = paths().plots_path() + 'Preprocessing/' + subjects.subject_id + '/'
    fname = f'{subjects.subject_id} Multipanel'
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







##

subject_code = exp_info.subjects_ids[-9]

subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
rt = subject.rt
plt.figure()
plt.hist(rt, bins=40, range=(0, 10), edgecolor='black', linewidth=1.2, density=False, stacked=True)
plt.title(subject_code)