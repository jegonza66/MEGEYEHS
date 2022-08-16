import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Paths
import Load

# Define file paths
behav_data_path = Paths.get().beh_path()
block_change_trials = [29,  59,  89, 119, 149, 179]
# block_change_trials = [29,  59]
##
subject = Load.subject('15912001')
# Load file
df_fran = subject.beh_data()

# Define colunms to keep and change block trials to drop
columns = ['Nstim', 'searchimage', 'target.started', 'fixation_target_2.started', 'search_img_2.started',
           'key_resp.keys', 'key_resp.rt', 'text_2.started']

t0_1 = 0

# Take text_2 values from next row
VS_end = df_fran.loc[~pd.isna(df_fran['text_2.started'])]['text_2.started'].reset_index(drop=True)
# Drop eyemap rows etc..
df_fran = df_fran.loc[~pd.isna(df_fran['target.started'])][columns].reset_index()
# Add text_2 values
df_fran['text_2.started'] = VS_end

missing_answers_fran = df_fran.loc[pd.isna(df_fran['key_resp.rt'])].reset_index(drop=True)

missed = []
for i in range(len(missing_answers_fran)):
    missed.append((missing_answers_fran["search_img_2.started"][i], missing_answers_fran["search_img_2.started"][i] + 10))
    exec('missed{} = missing_answers_fran["search_img_2.started"][{}]'.format(i,i))
    exec('missed{}_end = missed{} + 10'.format(i, i))

# Get trials with no answer
bad_trials_fran = df_fran.loc[pd.isna(df_fran['key_resp.rt'])]['searchimage']

## PLOT
# Missed responses on first session

t0_s1 = df_fran['search_img_2.started'].values[0] + df_fran['key_resp.rt'].values[0]
t0_s2 = 1907 + t0_s1

key_resp = [int(value) if value != 'None' else np.nan for value in df_fran['key_resp.keys'].values]
plt.figure()
plt.title('Behavioral data responses')
plt.plot(df_fran['search_img_2.started'].values + df_fran['key_resp.rt'].values - t0_s2, key_resp, 'o')
for i in range(len(missed)):
    plt.axvspan(missed[i][0] - t0_s2, missed[i][1] - t0_s2, 0, 4, color='grey', alpha=.5, label='Missed')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.grid()
plt.ylim([0, 5])
plt.xlim([0, df_fran['search_img_2.started'].values[-1] + df_fran['key_resp.rt'].values[-1] - t0_s2])

# plt.axvspan(missed2, missed2_end, 0, 4, color='grey', alpha=.5)
# plt.axvspan(missed3, missed3_end, 0, 4, color='grey', alpha=.5)
# plt.axvspan(missed4, missed4_end, 0, 4, color='grey', alpha=.5)
# plt.axvspan(missed5, missed5_end, 0, 4, color='grey', alpha=.5)
# plt.axvspan(missed6, missed6_end, 0, 4, color='grey', alpha=.5)
# plt.axvspan(missed7, missed7_end, 0, 4, color='grey', alpha=.5)
# plt.axvspan(missed8, missed8_end, 0, 4, color='grey', alpha=.5)


##

# Define time variables
df_fran['RT_total'] = df_fran['search_img_2.started'] + df_fran['key_resp.rt']
df_fran['fix2->VS'] = df_fran['search_img_2.started'] - df_fran['fixation_target_2.started']

MS_target_times_fran = np.array(df_fran['target.started'][1:])
RT_total_times_fran = np.array(df_fran['RT_total'][:-1])

ITI_fran = np.array(MS_target_times_fran - RT_total_times_fran)
ITI_fran = np.concatenate((ITI_fran, np.array([np.nan])))

rt_text2_ITI_fran = np.array(df_fran['text_2.started'] - df_fran['RT_total'])

text2_MS_ITI_fran = np.array(MS_target_times_fran - df_fran['text_2.started'][:-1])
text2_MS_ITI_fran = np.concatenate((text2_MS_ITI_fran, np.array([np.nan])))

df_fran['Total_ITI'] = ITI_fran
df_fran['rt_text2_ITI'] = df_fran['text_2.started'] - df_fran['RT_total']
df_fran['text2_MS_ITI'] = text2_MS_ITI_fran

ITI_fran = np.delete(arr=ITI_fran, obj=block_change_trials)
rt_text2_ITI_fran = np.delete(arr=rt_text2_ITI_fran, obj=block_change_trials)
text2_MS_ITI_fran = np.delete(arr=text2_MS_ITI_fran, obj=block_change_trials)


# PLOTS
fig, axs = plt.subplots(3, sharex=True)
fig.suptitle('ITI fran', fontsize=15)

axs[0].plot(ITI_fran)
axs[0].set_ylabel('Time (s)')
axs[0].set_title('RT -> MS')

axs[1].plot(rt_text2_ITI_fran)
axs[1].set_ylabel('Time (s)')
axs[1].set_title('RT -> text_2')

axs[2].plot(text2_MS_ITI_fran)
axs[2].set_ylabel('Time (s)')
axs[2].set_title('text_2 -> MS')

fig.tight_layout()


# Get data from each MSS
df_1_fran = df_fran.loc[df_fran['Nstim']==1]
df_2_fran = df_fran.loc[df_fran['Nstim']==2]
df_4_fran = df_fran.loc[df_fran['Nstim']==4]

df_1_fran['stDur'] = df_1_fran['fixation_target_2.started'] - df_1_fran['target.started']
df_2_fran['stDur'] = df_2_fran['fixation_target_2.started'] - df_2_fran['target.started']
df_4_fran['stDur'] = df_4_fran['fixation_target_2.started'] - df_4_fran['target.started']


# PLOTS
fig, axs = plt.subplots(4, sharex=True, figsize=(15, 9))
fig.suptitle('fran', fontsize=15)

axs[0].plot(df_fran['fix2->VS'])
axs[0].set_ylabel('Time (s)')
axs[0].set_title('fix2 -> VS')
axs[0].set_ylim([0, 2])

axs[1].plot(df_1_fran['stDur'])
axs[1].set_ylabel('Time (s)')
axs[1].set_title('stDur MSS=1')
axs[1].set_ylim([1, 3])

axs[2].plot(df_2_fran['stDur'])
axs[2].set_ylabel('Time (s)')
axs[2].set_title('stDur MSS=2')
axs[2].set_ylim([2.5, 4.5])

axs[3].plot(df_4_fran['stDur'])
axs[3].set_ylabel('Time (s)')
axs[3].set_title('stDur MSS=4')
axs[3].set_ylim([4, 6])

fig.tight_layout()

##
# Load suject
subject = Load.subject('11535009')
# Load file
df_test = subject.beh_data()
columns = ['Nstim', 'searchimage', 'corrAns_reminder_text_2.started', 'fixation_target.started', 'target.started', 'fixation_target_2.started', 'search_img_2.started', 'key_resp.rt']

# Get trials with answers only
df_test = df_test.loc[~pd.isna(df_test['key_resp.rt'])]
print('Missing anwsers: {}'.format(210-len(df_test)))
# Get response absolute times
df_test['RT_total'] = df_test['search_img_2.started'] + df_test['key_resp.rt']

Reminder_times_test = np.array(df_test['corrAns_reminder_text_2.started'][1:])
response_times = np.array(df_test['RT_total'][:-1])

rt_reminder = np.concatenate((Reminder_times_test - response_times, np.array([np.nan])))
reminder_fixation = (df_test['fixation_target.started'] - df_test['corrAns_reminder_text_2.started']).reset_index(drop=True).values
fixation_MS = df_test['target.started'] - df_test['fixation_target.started']
MS_fixation = df_test['fixation_target_2.started'] - df_test['target.started']

rt_reminder = np.delete(arr=rt_reminder, obj=block_change_trials)
reminder_fixation = np.delete(arr=reminder_fixation, obj=[i +1 for i in block_change_trials])

# PLOTS
fig, axs = plt.subplots(4, sharex=True)
fig.suptitle('{}\nTotal answers: {}'.format(subject.subject_id, len(df_test)), fontsize=15)

axs[0].plot(np.arange(len(rt_reminder)), rt_reminder)
axs[0].set_ylabel('Time (s)')
axs[0].set_title('Response -> Reminder')

axs[1].plot(np.arange(len(reminder_fixation)-1), reminder_fixation[1:])
axs[1].set_ylabel('Time (s)')
axs[1].set_title('Reminder -> fixation')

axs[2].plot(np.arange(len(df_test)), fixation_MS)
axs[2].set_ylabel('Time (s)')
axs[2].set_title('fixation -> MS')

axs[3].plot(np.arange(len(df_test)), MS_fixation)
axs[3].set_ylabel('Time (s)')
axs[3].set_title('MS -> fixation ')

fig.tight_layout()

##


behav_filename_test = 'automatic test 60_hybrid_search_builder_code_2022_Jun_07_1438.csv'
behav_filename_test = 'Auto_No_eyemap_hybrid_search_builder_code_2022_Jun_06_1715.csv'
columns = ['Nstim', 'searchimage', 'corrAns_reminder_text_2.started', 'fixation_target.started', 'target.started', 'fixation_target_2.started', 'search_img_2.started', 'text_2.started']


# Load file
df_test = pd.read_csv(behav_data_path+behav_filename_test)
# Take text_2 values from next row
VS_end = df_test.loc[~pd.isna(df_test['text_2.started'])]['text_2.started'].reset_index(drop=True)
# Drop eyemap rows etc..
df_test = df_test.loc[~pd.isna(df_test['target.started'])][columns].reset_index()
# Add text_2 values
df_test['text_2.started'] = VS_end


# Define time variables
df_test['fix2->VS'] = df_test['search_img_2.started'] - df_test['fixation_target_2.started']

Reminder_times_test = np.array(df_test['corrAns_reminder_text_2.started'][1:])
fixation_1_times_test = np.array(df_test['fixation_target.started'][1:])
MS_target_times_test = np.array(df_test['target.started'][1:])

Reminder_fixation_times_test = np.array(df_test['fixation_target.started'] - df_test['corrAns_reminder_text_2.started'])
fixation_MS_times_test = np.array(df_test['target.started'] - df_test['fixation_target.started'])

text2_Reminder_ITI_test = np.array(Reminder_times_test - df_test['text_2.started'][:-1])
text2_Reminder_ITI_test = np.concatenate((text2_Reminder_ITI_test, np.array([np.nan])))

text2_MS_ITI_test = np.array(MS_target_times_test - df_test['text_2.started'][:-1])
text2_MS_ITI_test = np.concatenate((text2_MS_ITI_test, np.array([np.nan])))

df_test['text2_MS_ITI'] = text2_MS_ITI_test

text2_MS_ITI_test = np.delete(arr=text2_MS_ITI_test, obj=block_change_trials)
text2_Reminder_ITI_test = np.delete(arr=text2_Reminder_ITI_test, obj=block_change_trials)

# PLOTS
fig, axs = plt.subplots(4, sharex=True)
fig.suptitle('ITI test', fontsize=15)

axs[0].plot(text2_MS_ITI_test)
axs[0].set_ylabel('Time (s)')
axs[0].set_title('text_2 -> MS')

axs[1].plot(text2_Reminder_ITI_test)
axs[1].set_ylabel('Time (s)')
axs[1].set_title('text_2 -> Reminder')

axs[2].plot(Reminder_fixation_times_test)
axs[2].set_ylabel('Time (s)')
axs[2].set_title('Reminder -> fixation')

axs[3].plot(fixation_MS_times_test)
axs[3].set_ylabel('Time (s)')
axs[3].set_title('fixation -> MS')

fig.tight_layout()

## Minimal
behav_filename_test = 'automatic test 60_hybrid_search_builder_code_2022_Jun_07_1438.csv'
behav_filename_test = 'minimal_comp_hybrid_search_builder_code_win_ET_eyemap_SPMIC_UoN_MEG_Automatic_EYEMAP_minimal_component_2022_Jun_08_1114.csv'
columns = ['Nstim', 'searchimage', 'corrAns_reminder_text_2.started', 'fixation_target.started', 'target.started', 'fixation_target_2.started', 'search_img_2.started', 'text_2.started']


# Load file
df_test = pd.read_csv(behav_data_path+behav_filename_test)
# Take text_2 values from next row
VS_end = df_test.loc[~pd.isna(df_test['text_2.started'])]['text_2.started'].reset_index(drop=True)
# Drop eyemap rows etc..
df_test = df_test.loc[~pd.isna(df_test['target.started'])][columns].reset_index()
# Add text_2 values
df_test['text_2.started'] = VS_end


# Define time variables
df_test['fix2->VS'] = df_test['search_img_2.started'] - df_test['fixation_target_2.started']

Reminder_times_test = np.array(df_test['corrAns_reminder_text_2.started'][1:])
fixation_1_times_test = np.array(df_test['fixation_target.started'][1:])
MS_target_times_test = np.array(df_test['target.started'][1:])

Reminder_fixation_times_test = np.array(df_test['fixation_target.started'] - df_test['corrAns_reminder_text_2.started'])
fixation_MS_times_test = np.array(df_test['target.started'] - df_test['fixation_target.started'])

text2_Reminder_ITI_test = np.array(Reminder_times_test - df_test['text_2.started'][:-1])
text2_Reminder_ITI_test = np.concatenate((text2_Reminder_ITI_test, np.array([np.nan])))

text2_MS_ITI_test = np.array(MS_target_times_test - df_test['text_2.started'][:-1])
text2_MS_ITI_test = np.concatenate((text2_MS_ITI_test, np.array([np.nan])))

df_test['text2_MS_ITI'] = text2_MS_ITI_test

text2_MS_ITI_test = np.delete(arr=text2_MS_ITI_test, obj=block_change_trials)
text2_Reminder_ITI_test = np.delete(arr=text2_Reminder_ITI_test, obj=block_change_trials)

# PLOTS
fig, axs = plt.subplots(4, sharex=True)
fig.suptitle('ITI test', fontsize=15)

axs[0].plot(text2_MS_ITI_test)
axs[0].set_ylabel('Time (s)')
axs[0].set_title('text_2 -> MS')

axs[1].plot(text2_Reminder_ITI_test)
axs[1].set_ylabel('Time (s)')
axs[1].set_title('text_2 -> Reminder')

axs[2].plot(Reminder_fixation_times_test)
axs[2].set_ylabel('Time (s)')
axs[2].set_title('Reminder -> fixation')

axs[3].plot(fixation_MS_times_test)
axs[3].set_ylabel('Time (s)')
axs[3].set_title('fixation -> MS')

fig.tight_layout()


##
behav_filename_fran = '15912001_hybrid_search_builder_code_2022_May_31_1009.csv'
columns = ['Nstim', 'searchimage', 'target.started', 'fixation_target_2.started', 'search_img_2.started', 'key_resp.rt', 'text_2.started']

# Load file
df_fran = pd.read_csv(behav_data_path+behav_filename_fran)
# Take text_2 values from next row
VS_end = df_fran.loc[~pd.isna(df_fran['text_2.started'])]['text_2.started'].reset_index(drop=True)
# Drop eyemap rows etc..
df_fran = df_fran.loc[~pd.isna(df_fran['target.started'])][columns].reset_index()
# Add text_2 values
df_fran['text_2.started'] = VS_end

# Get trials with no answer
bad_trials_fran = df_fran.loc[pd.isna(df_fran['key_resp.rt'])]['searchimage']

# Define time variables
df_fran['RT_total'] = df_fran['search_img_2.started'] + df_fran['key_resp.rt']
df_fran['fix2->VS'] = df_fran['search_img_2.started'] - df_fran['fixation_target_2.started']

MS_target_times_fran = np.array(df_fran['target.started'][1:])
RT_total_times_fran = np.array(df_fran['RT_total'][:-1])

ITI_fran = np.array(MS_target_times_fran - RT_total_times_fran)
ITI_fran = np.concatenate((ITI_fran, np.array([np.nan])))

rt_text2_ITI_fran = np.array(df_fran['text_2.started'] - df_fran['RT_total'])

text2_MS_ITI_fran = np.array(MS_target_times_fran - df_fran['text_2.started'][:-1])
text2_MS_ITI_fran = np.concatenate((text2_MS_ITI_fran, np.array([np.nan])))

df_fran['Total_ITI'] = ITI_fran
df_fran['rt_text2_ITI'] = df_fran['text_2.started'] - df_fran['RT_total']
df_fran['text2_MS_ITI'] = text2_MS_ITI_fran

ITI_fran = np.delete(arr=ITI_fran, obj=block_change_trials)
rt_text2_ITI_fran = np.delete(arr=rt_text2_ITI_fran, obj=block_change_trials)
text2_MS_ITI_fran = np.delete(arr=text2_MS_ITI_fran, obj=block_change_trials)


# PLOTS
fig, axs = plt.subplots(3, sharex=True)
fig.suptitle('ITI fran', fontsize=15)

axs[0].plot(ITI_fran)
axs[0].set_ylabel('Time (s)')
axs[0].set_title('RT -> MS')

axs[1].plot(rt_text2_ITI_fran)
axs[1].set_ylabel('Time (s)')
axs[1].set_title('RT -> text_2')

axs[2].plot(text2_MS_ITI_fran)
axs[2].set_ylabel('Time (s)')
axs[2].set_title('text_2 -> MS')

fig.tight_layout()


# Get data from each MSS
df_1_fran = df_fran.loc[df_fran['Nstim']==1]
df_2_fran = df_fran.loc[df_fran['Nstim']==2]
df_4_fran = df_fran.loc[df_fran['Nstim']==4]

df_1_fran['stDur'] = df_1_fran['fixation_target_2.started'] - df_1_fran['target.started']
df_2_fran['stDur'] = df_2_fran['fixation_target_2.started'] - df_2_fran['target.started']
df_4_fran['stDur'] = df_4_fran['fixation_target_2.started'] - df_4_fran['target.started']


# PLOTS
fig, axs = plt.subplots(4, sharex=True, figsize=(15, 9))
fig.suptitle('fran', fontsize=15)

axs[0].plot(df_fran['fix2->VS'])
axs[0].set_ylabel('Time (s)')
axs[0].set_title('fix2 -> VS')
axs[0].set_ylim([0, 2])

axs[1].plot(df_1_fran['stDur'])
axs[1].set_ylabel('Time (s)')
axs[1].set_title('stDur MSS=1')
axs[1].set_ylim([1, 3])

axs[2].plot(df_2_fran['stDur'])
axs[2].set_ylabel('Time (s)')
axs[2].set_title('stDur MSS=2')
axs[2].set_ylim([2.5, 4.5])

axs[3].plot(df_4_fran['stDur'])
axs[3].set_ylabel('Time (s)')
axs[3].set_title('stDur MSS=4')
axs[3].set_ylim([4, 6])

fig.tight_layout()


##
behav_filename_eeg = '533569_hybrid_search_builder_code_2022_Feb_28_1015.csv'

eeg_columns = ['Nstim', 'searchimage', 'target.started', 'fixation_target.started', 'search_img.started', 'key_resp.rt', 'text_2.started']

# Load file
df_eeg = pd.read_csv(behav_data_path+behav_filename_eeg)
# Take text_2 values from next row
VS_end = df_eeg.loc[~pd.isna(df_eeg['text_2.started'])]['text_2.started'].reset_index(drop=True)
# Drop eyemap rows etc..
df_eeg = df_eeg.loc[~pd.isna(df_eeg['target.started'])][eeg_columns].reset_index()
# Add text_2 values
df_eeg['text_2.started'] = VS_end

# Get trials with no answer
bad_trials_eeg_eeg = df_eeg.loc[pd.isna(df_eeg['key_resp.rt'])]['searchimage']

# Define time variables
df_eeg['RT_total'] = df_eeg['search_img.started'] + df_eeg['key_resp.rt']
df_eeg['fix2->VS'] = df_eeg['search_img.started'] - df_eeg['fixation_target.started']

MS_target_times_eeg = np.array(df_eeg['target.started'][1:])
RT_total_times_eeg = np.array(df_eeg['RT_total'][:-1])

ITI_eeg = np.array(MS_target_times_eeg - RT_total_times_eeg)
ITI_eeg = np.concatenate((ITI_eeg, np.array([np.nan])))

rt_text2_ITI_eeg = np.array(df_eeg['text_2.started'] - df_eeg['RT_total'])

text2_MS_ITI_eeg = np.array(MS_target_times_eeg - df_eeg['text_2.started'][:-1])
text2_MS_ITI_eeg = np.concatenate((text2_MS_ITI_eeg, np.array([np.nan])))

df_eeg['Total_ITI'] = ITI_eeg
df_eeg['rt_text2_ITI'] = df_eeg['text_2.started'] - df_eeg['RT_total']
df_eeg['text2_MS_ITI'] = text2_MS_ITI_eeg

ITI_eeg = np.delete(arr=ITI_eeg, obj=block_change_trials)
rt_text2_ITI_eeg = np.delete(arr=rt_text2_ITI_eeg, obj=block_change_trials)
text2_MS_ITI_eeg = np.delete(arr=text2_MS_ITI_eeg, obj=block_change_trials)


# PLOTS
fig, axs = plt.subplots(3, sharex=True)
fig.suptitle('ITI eeg_eeg', fontsize=15)

axs[0].plot(ITI_eeg)
axs[0].set_ylabel('Time (s)')
axs[0].set_title('RT -> MS')

axs[1].plot(rt_text2_ITI_eeg)
axs[1].set_ylabel('Time (s)')
axs[1].set_title('RT -> text_2')

axs[2].plot(text2_MS_ITI_eeg)
axs[2].set_ylabel('Time (s)')
axs[2].set_title('text_2 -> MS')

fig.tight_layout()


# Get data from each MSS
df_1_eeg = df_eeg.loc[df_eeg['Nstim']==1]
df_2_eeg = df_eeg.loc[df_eeg['Nstim']==2]
df_4_eeg = df_eeg.loc[df_eeg['Nstim']==4]

df_1_eeg['stDur'] = df_1_eeg['fixation_target.started'] - df_1_eeg['target.started']
df_2_eeg['stDur'] = df_2_eeg['fixation_target.started'] - df_2_eeg['target.started']
df_4_eeg['stDur'] = df_4_eeg['fixation_target.started'] - df_4_eeg['target.started']


# PLOTS
fig, axs = plt.subplots(4, sharex=True, figsize=(15, 9))
fig.suptitle('eeg', fontsize=15)

axs[0].plot(df_eeg['fix2->VS'])
axs[0].set_ylabel('Time (s)')
axs[0].set_title('fix2 -> VS')
axs[0].set_ylim([0, 2])

axs[1].plot(df_1_eeg['stDur'])
axs[1].set_ylabel('Time (s)')
axs[1].set_title('stDur MSS=1')
axs[1].set_ylim([1, 3])

axs[2].plot(df_2_eeg['stDur'])
axs[2].set_ylabel('Time (s)')
axs[2].set_title('stDur MSS=2')
axs[2].set_ylim([2.5, 4.5])

axs[3].plot(df_4_eeg['stDur'])
axs[3].set_ylabel('Time (s)')
axs[3].set_title('stDur MSS=4')
axs[3].set_ylim([4, 6])

fig.tight_layout()


##
behav_filename_auto_eeg = 'automatic test 60_hybrid_search_builder_code_2022_Jun_07_1438.csv'

columns = ['Nstim', 'searchimage', 'target.started', 'fixation_target_2.started', 'search_img_2.started', 'text_2.started']

# Load file
df_auto_eeg = pd.read_csv(behav_data_path+behav_filename_auto_eeg)
# Take text_2 values from next row
VS_end = df_auto_eeg.loc[~pd.isna(df_auto_eeg['text_2.started'])]['text_2.started'].reset_index(drop=True)
# Drop eyemap rows etc..
df_auto_eeg = df_auto_eeg.loc[~pd.isna(df_auto_eeg['target.started'])][columns].reset_index()
# Add text_2 values
df_auto_eeg['text_2.started'] = VS_end

index_1 = (df_auto_eeg['Nstim']==1).values
index_2 = (df_auto_eeg['Nstim']==2).values
index_4 = (df_auto_eeg['Nstim']==4).values
auto_rt = np.zeros(len(df_auto_eeg))
auto_rt[index_1] = 0.2
auto_rt[index_2] = 0.35
auto_rt[index_4] = 0.5

df_auto_eeg['key_resp.rt'] = auto_rt

# Get trials with no answer
bad_trials_auto_eeg_auto_eeg = df_auto_eeg.loc[pd.isna(df_auto_eeg['key_resp.rt'])]['searchimage']

# Define time variables
df_auto_eeg['RT_total'] = df_auto_eeg['search_img_2.started'] + df_auto_eeg['key_resp.rt']
df_auto_eeg['fix2->VS'] = df_auto_eeg['search_img_2.started'] - df_auto_eeg['fixation_target_2.started']

MS_target_times_test = np.array(df_auto_eeg['target.started'][1:])
RT_total_times_test = np.array(df_auto_eeg['RT_total'][:-1])

ITI_test = np.array(MS_target_times_test - RT_total_times_test)
ITI_test = np.concatenate((ITI_test, np.array([np.nan])))

rt_text2_ITI_test = np.array(df_auto_eeg['text_2.started'] - df_auto_eeg['RT_total'])

text2_MS_ITI_test = np.array(MS_target_times_test - df_auto_eeg['text_2.started'][:-1])
text2_MS_ITI_test = np.concatenate((text2_MS_ITI_test, np.array([np.nan])))

df_auto_eeg['Total_ITI'] = ITI_test
df_auto_eeg['rt_text2_ITI'] = df_auto_eeg['text_2.started'] - df_auto_eeg['RT_total']
df_auto_eeg['text2_MS_ITI'] = text2_MS_ITI_test

ITI_test = np.delete(arr=ITI_test, obj=block_change_trials)
rt_text2_ITI_test = np.delete(arr=rt_text2_ITI_test, obj=block_change_trials)
text2_MS_ITI_test = np.delete(arr=text2_MS_ITI_test, obj=block_change_trials)


# PLOTS
fig, axs = plt.subplots(3, sharex=True)
fig.suptitle('ITI test', fontsize=15)

axs[0].plot(ITI_test)
axs[0].set_ylabel('Time (s)')
axs[0].set_title('RT -> MS')

axs[1].plot(rt_text2_ITI_test)
axs[1].set_ylabel('Time (s)')
axs[1].set_title('RT -> text_2')

axs[2].plot(text2_MS_ITI_test)
axs[2].set_ylabel('Time (s)')
axs[2].set_title('text_2 -> MS')

fig.tight_layout()


# Get data from each MSS
df_1_test = df_auto_eeg.loc[df_auto_eeg['Nstim']==1]
df_2_test = df_auto_eeg.loc[df_auto_eeg['Nstim']==2]
df_4_test = df_auto_eeg.loc[df_auto_eeg['Nstim']==4]

df_1_test['stDur'] = df_1_test['fixation_target_2.started'] - df_1_test['target.started']
df_2_test['stDur'] = df_2_test['fixation_target_2.started'] - df_2_test['target.started']
df_4_test['stDur'] = df_4_test['fixation_target_2.started'] - df_4_test['target.started']


# PLOTS
fig, axs = plt.subplots(4, sharex=True, figsize=(15, 9))
fig.suptitle('test', fontsize=15)

axs[0].plot(df_auto_eeg['fix2->VS'])
axs[0].set_ylabel('Time (s)')
axs[0].set_title('fix2 -> VS')
axs[0].set_ylim([0, 2])

axs[1].plot(df_1_test['stDur'])
axs[1].set_ylabel('Time (s)')
axs[1].set_title('stDur MSS=1')
axs[1].set_ylim([1, 3])

axs[2].plot(df_2_test['stDur'])
axs[2].set_ylabel('Time (s)')
axs[2].set_title('stDur MSS=2')
axs[2].set_ylim([2.5, 4.5])

axs[3].plot(df_4_test['stDur'])
axs[3].set_ylabel('Time (s)')
axs[3].set_title('stDur MSS=4')
axs[3].set_ylim([4, 6])

fig.tight_layout()


##
behav_filename_auto_eeg = 'Auto_loop_off_hybrid_search_builder_code_2022_Jun_06_1442.csv'

columns = ['Nstim', 'searchimage', 'target.started', 'fixation_target_2.started', 'search_img_2.started', 'emap_screen_2.started']

# Load file
df_auto_eeg = pd.read_csv(behav_data_path+behav_filename_auto_eeg)
# Take text_2 values from next row
VS_end = df_auto_eeg.loc[~pd.isna(df_auto_eeg['emap_screen_2.started'])]['emap_screen_2.started'].reset_index(drop=True)
# Drop eyemap rows etc..
df_auto_eeg = df_auto_eeg.loc[~pd.isna(df_auto_eeg['target.started'])][columns].reset_index()
# Add text_2 values
df_auto_eeg['emap_screen_2.started'] = VS_end

index_1 = (df_auto_eeg['Nstim']==1).values
index_2 = (df_auto_eeg['Nstim']==2).values
index_4 = (df_auto_eeg['Nstim']==4).values
auto_rt = np.zeros(len(df_auto_eeg))
auto_rt[index_1] = 0.2
auto_rt[index_2] = 0.35
auto_rt[index_4] = 0.5

df_auto_eeg['key_resp.rt'] = auto_rt

# Get trials with no answer
bad_trials_auto_eeg_auto_eeg = df_auto_eeg.loc[pd.isna(df_auto_eeg['key_resp.rt'])]['searchimage']

# Define time variables
df_auto_eeg['RT_total'] = df_auto_eeg['search_img_2.started'] + df_auto_eeg['key_resp.rt']
df_auto_eeg['fix2->VS'] = df_auto_eeg['search_img_2.started'] - df_auto_eeg['fixation_target_2.started']

MS_target_times_test = np.array(df_auto_eeg['target.started'][1:])
RT_total_times_test = np.array(df_auto_eeg['RT_total'][:-1])

ITI_test = np.array(MS_target_times_test - RT_total_times_test)
ITI_test = np.concatenate((ITI_test, np.array([np.nan])))

rt_text2_ITI_test = np.array(df_auto_eeg['emap_screen_2.started'] - df_auto_eeg['RT_total'])

text2_MS_ITI_test = np.array(MS_target_times_test - df_auto_eeg['emap_screen_2.started'][:-1])
text2_MS_ITI_test = np.concatenate((text2_MS_ITI_test, np.array([np.nan])))

df_auto_eeg['Total_ITI'] = ITI_test
df_auto_eeg['rt_text2_ITI'] = df_auto_eeg['emap_screen_2.started'] - df_auto_eeg['RT_total']
df_auto_eeg['text2_MS_ITI'] = text2_MS_ITI_test

ITI_test = np.delete(arr=ITI_test, obj=block_change_trials)
rt_text2_ITI_test = np.delete(arr=rt_text2_ITI_test, obj=block_change_trials)
text2_MS_ITI_test = np.delete(arr=text2_MS_ITI_test, obj=block_change_trials)


# PLOTS
fig, axs = plt.subplots(3, sharex=True)
fig.suptitle('ITI test', fontsize=15)

axs[0].plot(ITI_test)
axs[0].set_ylabel('Time (s)')
axs[0].set_title('RT -> MS')

axs[1].plot(rt_text2_ITI_test)
axs[1].set_ylabel('Time (s)')
axs[1].set_title('RT -> text_2')

axs[2].plot(text2_MS_ITI_test)
axs[2].set_ylabel('Time (s)')
axs[2].set_title('text_2 -> MS')

fig.tight_layout()