import pandas as pd
import plot_preproc
from paths import paths
import setup
import load
import save
import matplotlib.pyplot as plt


save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

all_fixations = pd.DataFrame()
all_saccades = pd.DataFrame()
all_bh_data = pd.DataFrame()
all_rt = pd.DataFrame()
all_corr_ans = pd.DataFrame()

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

    all_fixations = pd.concat([all_fixations, fixations])
    all_saccades = pd.concat([all_saccades, saccades])
    all_bh_data = pd.concat([all_bh_data, bh_data])
    all_rt = pd.concat([all_rt, rt])
    all_corr_ans = pd.concat([all_corr_ans, corr_ans])

# Define all subjects class instance
subjects = setup.all_subjects(all_fixations, all_saccades, all_bh_data, all_rt, all_corr_ans)

plot_preproc.first_fixation_delay(subject=subjects)
plot_preproc.pupil_size_increase(subject=subjects)


plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplots(2, 2, figsize=(12, 7))

plot_preproc.fixation_duration(subject=subjects, ax=axs[0, 0])
plot_preproc.saccades_amplitude(subject=subjects, ax=axs[0, 1])
plot_preproc.saccades_dir_hist(subject=subjects, fig=fig, ax=axs[1, 0])
plot_preproc.sac_main_seq(subject=subjects, ax=axs[1, 1])

fig.tight_layout()

save_path = paths().plots_path() + 'Preprocessing/' + subjects.subject_id + '/'
fname = f'{subjects.subject_id} Multipanel'
save.fig(fig=fig, path=save_path, fname=fname)


