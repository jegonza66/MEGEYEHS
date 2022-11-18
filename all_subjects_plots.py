import pandas as pd
import preproc_plot
from paths import paths
import setup
import load
import numpy as np


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

class all_subjects:

    def __init__(self, all_fixations, all_saccades, all_bh_data, all_rt):
        self.subject_id = 'All_Subjects'
        self.fixations = all_fixations
        self.saccades = all_saccades
        self.trial = np.arange(1, 211)
        self.bh_data = all_bh_data
        self.rt = all_rt
        self.corr_ans = all_corr_ans


subjects = all_subjects(all_fixations, all_saccades, all_bh_data, all_rt)

preproc_plot.first_fixation_delay(subject=subjects)
preproc_plot.pupil_size_increase(subject=subjects)
# preproc_plot.performance(subject=subjects)
preproc_plot.fixation_duration(subject=subjects)
preproc_plot.saccades_amplitude(subject=subjects)
preproc_plot.saccades_dir_hist(subject=subjects)
preproc_plot.sac_main_seq(subject=subjects)



