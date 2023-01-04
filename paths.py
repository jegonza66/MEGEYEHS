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


    def config_path(self):
        """
        Paths to run configuration.

        Returns
        -------
        config_path: str
            Path in str format to the config directory.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n':  # Asus Rog
            config_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/Python/Config/'
        elif self.name == 'usuario\n':  # Liaa Colores
            config_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/Python/Config/'
        else:  # Notts
            config_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/Config/'
        return config_path

    def ctf_path(self):
        """
        Paths to participants MEG data.

        Returns
        -------
        ctf_path: str
            Path in str format to the CTF folder containing the MEG data.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n': # Asus Rog
            ctf_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/DATA/CTF_DATA/'
        elif self.name == 'usuario\n': # Liaa Colores
            ctf_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/DATA/CTF_DATA/'
        else: # Notts
            ctf_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/DATA/CTF_DATA/'
        return ctf_path


    def et_path(self):
        """
        Paths to participants Eye-Tracker data.

        Returns
        -------
        et_path: str
            Path in str format to the ET folder containing the Eye-Tracker data.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            et_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/DATA/ET_DATA/'
        elif self.name == 'usuario\n': # Liaa Colores
            et_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/DATA/ET_DATA/'
        else:
            et_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/DATA/ET_DATA/'
        return et_path


    def bh_path(self):
        """
        Paths to participants behavioural data.

        Returns
        -------
        beh_path: str
            Path in str format to the BH folder containing the behavioural data.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            bh_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/DATA/BH_DATA/'
        elif self.name == 'usuario\n': # Liaa Colores
            bh_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/DATA/BH_DATA/'
        else:
            bh_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/DATA/BH_DATA/'
        return bh_path


    def mri_path(self):
        """
        Paths to participants MRI data.

        Returns
        -------
        mri_path: str
            Path in str format to the MRI folder containing the MRI data.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            mri_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/DATA/MRI_DATA/'
        elif self.name == 'usuario\n': # Liaa Colores
            mri_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/DATA/MRI_DATA/'
        else:
            mri_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/DATA/MRI_DATA/'
        return mri_path


    def opt_path(self):
        """
        Paths to participants digitalization data.

        Returns
        -------
        opt_path: str
            Path in str format to the OPT folder containing the digitalization data.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            opt_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/DATA/OPT_DATA/'
        elif self.name == 'usuario\n': # Liaa Colores
            opt_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/DATA/OPT_DATA/'
        else:
            opt_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/DATA/OPT_DATA/'
        return opt_path


    def save_path(self):
        """
        Paths to the Saves data folder.

        Returns
        -------
        save_path: str
            Path in str format to the Saves folder.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            save_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/Python/Save/'
        elif self.name == 'usuario\n':  # Liaa Colores
            save_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/Python/Save/'
        else:
            save_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/Save/'

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

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            preproc_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/Python/Save/Preprocessed_Data/'
        elif self.name == 'usuario\n': # Liaa Colores
            preproc_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/Python/Save/Preprocessed_Data/'
        else:
            preproc_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/Save/Preprocessed_Data/'

        # Create directory if it doesn't exist
        os.makedirs(preproc_path, exist_ok=True)

        return preproc_path


    def filtered_path(self):
        """
        Paths to the filtered data folder.

        Returns
        -------
        filtered_path: str
            Path in str format to the folder containing the filtered data.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            filtered_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/Python/Save/Filtered_Data/'
        elif self.name == 'usuario\n': # Liaa Colores
            filtered_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/Python/Save/Filtered_Data/'
        else:
            filtered_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/Save/Filtered_Data/'

        # Create directory if it doesn't exist
        os.makedirs(filtered_path, exist_ok=True)

        return filtered_path


    def ica_path(self):
        """
        Paths to the filtered data folder.

        Returns
        -------
        filtered_path: str
            Path in str format to the folder containing the filtered data.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            ica_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/Python/Save/ICA_Data/'
        elif self.name == 'usuario\n': # Liaa Colores
            ica_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/Python/Save/ICA_Data/'
        else:
            ica_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/Save/ICA_Data/'

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

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            results_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/Python/Results/'
        elif self.name == 'usuario\n': # Liaa Colores
            results_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/Python/Results/'
        else:
            results_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/Results/'

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

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            plots_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/Python/Plots/'
        elif self.name == 'usuario\n': # Liaa Colores
            plots_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/Python/Plots/'
        else:
            plots_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/Plots/'

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

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            item_pos_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/DATA/pos_items_210_target.csv'
        elif self.name == 'usuario\n': # Liaa Colores
            item_pos_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/DATA/pos_items_210_target.csv'
        else:
            item_pos_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/DATA/pos_items_210_target.csv'

        return item_pos_path


    def experiment_path(self):
        """
        Paths to the Psychopy experiment directory.

        Returns
        -------
        exp_path: str
            Path in str format to the items positions file.
        """

        if self.name == 'laptop-5i5qsv76\\joaco\n':
            exp_path = 'C:/Users/joaco/OneDrive - The University of Nottingham/MEGEYEHS/Psychopy_Experiment/'
        elif self.name == 'usuario\n': # Liaa Colores
            exp_path = '/mnt/6a6fd40a-e256-4844-8004-0e60d95969e8/MEGEYEHS/Psychopy_Experiment/'
        else:
            exp_path = 'C:/Users/lpajg1/OneDrive - The University of Nottingham/MEGEYEHS/Psychopy_Experiment/'

        return exp_path