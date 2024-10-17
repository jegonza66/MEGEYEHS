import os

class paths:
    """
    Paths to participants data.

    Attributes
    -------
    name: str
        Runing computers name result of os.popen('whoami').read().
    """

    def __init__(self):
        self.name = os.popen('whoami').read()

        if self.name == 'laptop-5i5qsv76\\joaco\n':  # Asus Rog
            self.main_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/'

        elif self.name == 'desktop-r5hd7ct\\joac\n':  # Liaa Colores
            self.main_path = 'D:/OneDrive - The University of Nottingham/MEGEYEHS/'

        elif self.name == 'desktop-59r7a1d\\usuario\n':  # Desktop
            self.main_path = 'C:/Users/Usuario/OneDrive - The University of Nottingham/MEGEYEHS/'

        elif self.name == 'ad\\lpajg1\n':  # Notts
            self.main_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/'

        else:
            raise ValueError(f'No paths set for pc: {self.name}')


    def config_path(self):
        """
        Paths to run configuration.

        Returns
        -------
        config_path: str
            Path in str format to the config directory.
        """

        config_path = self.main_path + 'Python/Config/'

        return config_path


    def ctf_path(self):
        """
        Paths to participants MEG data.

        Returns
        -------
        ctf_path: str
            Path in str format to the CTF folder containing the MEG data.
        """

        ctf_path = self.main_path + 'DATA/CTF_DATA/'

        return ctf_path


    def et_path(self):
        """
        Paths to participants Eye-Tracker data.

        Returns
        -------
        et_path: str
            Path in str format to the ET folder containing the Eye-Tracker data.
        """

        et_path = self.main_path + 'DATA/ET_DATA/'

        return et_path


    def bh_path(self):
        """
        Paths to participants behavioural data.

        Returns
        -------
        beh_path: str
            Path in str format to the BH folder containing the behavioural data.
        """

        bh_path = self.main_path + 'DATA/BH_DATA/'

        return bh_path


    def mri_path(self):
        """
        Paths to participants MRI data.

        Returns
        -------
        mri_path: str
            Path in str format to the MRI folder containing the MRI data.
        """

        mri_path = self.main_path + 'DATA/MRI_DATA/'

        return mri_path


    def opt_path(self):
        """
        Paths to participants digitalization data.

        Returns
        -------
        opt_path: str
            Path in str format to the OPT folder containing the digitalization data.
        """

        opt_path = self.main_path + 'DATA/OPT_DATA/'

        return opt_path


    def save_path(self):
        """
        Paths to the Saves data folder.

        Returns
        -------
        save_path: str
            Path in str format to the Saves folder.
        """

        save_path = self.main_path + 'Python/Save/'

        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        return save_path


    def preproc_path(self):
        """
        Paths to the preprocessed data folder.

        Returns
        -------
        preproc_path: str
            Path in str format to the folder containing the preprocessed data.
        """

        preproc_path = self.main_path + 'Python/Save/Preprocessed_Data/'

        # Create directory if it doesn't exist
        os.makedirs(preproc_path, exist_ok=True)

        return preproc_path


    def filtered_path_raw(self):
        """
        Paths to the filtered data folder.

        Returns
        -------
        filtered_path: str
            Path in str format to the folder containing the filtered data.
        """

        filtered_path = self.main_path + 'Python/Save/Filtered_Data_RAW/'

        # Create directory if it doesn't exist
        os.makedirs(filtered_path, exist_ok=True)

        return filtered_path

    def filtered_path_ica(self):
        """
        Paths to the filtered data folder.

        Returns
        -------
        filtered_path: str
            Path in str format to the folder containing the filtered data.
        """

        filtered_path = self.main_path + 'Python/Save/Filtered_Data_ICA/'

        # Create directory if it doesn't exist
        os.makedirs(filtered_path, exist_ok=True)

        return filtered_path


    def ica_path(self):
        """
        Paths to the ica clean data folder.

        Returns
        -------
        filtered_path: str
            Path in str format to the folder containing the ica clean data.
        """

        ica_path = self.main_path + 'Python/Save/ICA_Data/'

        # Create directory if it doesn't exist
        os.makedirs(ica_path, exist_ok=True)

        return ica_path


    def results_path(self):
        """
        Paths to the results folder.

        Returns
        -------
        results_path: str
            Path in str format to the folder to store the results.
        """

        results_path = self.main_path + 'Python/Results/'

        # Create directory if it doesn't exist
        os.makedirs(results_path, exist_ok=True)

        return results_path


    def plots_path(self):
        """
        Paths to the plots folder.

        Returns
        -------
        plots_path: str
            Path in str format to the folder to store the results.
        """

        plots_path = self.main_path + 'Python/Plots/'

        # Create directory if it doesn't exist
        os.makedirs(plots_path, exist_ok=True)

        return plots_path


    def item_pos_path(self):
        """
        Paths to the search items positions file.

        Returns
        -------
        item_pos_path: str
            Path in str format to the items positions file.
        """

        item_pos_path = self.main_path + 'DATA/pos_items_210_target.csv'

        return item_pos_path


    def experiment_path(self):
        """
        Paths to the Psychopy experiment directory.

        Returns
        -------
        exp_path: str
            Path in str format to the items positions file.
        """

        exp_path = self.main_path + 'Psychopy_Experiment/'

        return exp_path


    def sources_path(self):
        """
        Paths to the directory containing saved data for source estimation. Such as source model, bem model, forward model.

        Returns
        -------
        exp_path: str
            Path in str format to the source directory.
        """

        sources_path = self.main_path + 'Python/Save/Source_Data/'

        os.makedirs(sources_path, exist_ok=True)

        return sources_path
    

    def fwd_path(self, subject, subject_code, ico, spacing, surf_vol):

        sources_path_subject = self.sources_path() + subject.subject_id
        
        # Load forward model
        if surf_vol == 'volume':
            fname_fwd = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}-fwd.fif'
        elif surf_vol == 'surface':
            fname_fwd = sources_path_subject + f'/{subject_code}_surface_ico{ico}-fwd.fif'
        elif surf_vol == 'mixed':
            fname_fwd = sources_path_subject + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}-fwd.fif'
        
        return fname_fwd
    
    def filter_path(self, subject, subject_code, ico, spacing, surf_vol, pick_ori, model_name):
        
        sources_path_subject = self.sources_path() + subject.subject_id
        
        # Load filter
        if surf_vol == 'volume':
            fname_filter = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}_{pick_ori}-{model_name}.fif'
        elif surf_vol == 'surface':
            fname_filter = sources_path_subject + f'/{subject_code}_surface_ico{ico}_{pick_ori}-{model_name}.fif'
        elif surf_vol == 'mixed':
            fname_filter = sources_path_subject + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}_{pick_ori}-{model_name}.fif'

        return fname_filter