import os
import pandas as pd
import setup
from paths import paths

exp_info = setup.exp_info()
bh_path = paths().bh_path()

# Parse the log files and get the images that were shown in each trial, in the correct order
def parse_images(log_file):
    file = open(log_file, "r")
    lines = file.readlines()
    images = []
    positions_X = []
    positions_Y = []
    for index, line in enumerate(lines):
        if ".jpg" in line or ".png" in line:
            if " = " in line:
                # Delete 'memstim/ from line.split(" = ")[1]
                # Delete '\n from line.split(" = ")[1]
                image_name = line.split(" = ")[1].split("'\n")[0]
                if 'memstim' in image_name:
                    image_name = image_name.split('memstim/')[1]
                    position = lines[index-2].split("array([")[1].split("]")[0]
                    #I want to turn the string into a list of floats
                    position = list(map(float, position.split(",")))
                    positions_X.append(position[0] + 1920/2)
                    positions_Y.append(-position[1] + 1080/2)
                else:
                    image_name = image_name.split('\'')[1]
                images.append(image_name)
    # Group the images and their respective coordinates by trial in a dictionary
    # Keep the order of the images and the coordinates
    # Every 6 images there is a new trial and every 5 coordinates there is a new trial
    trials = {}
    trial_number = 1
    for i in range(0, len(positions_X), 5):
        trials[trial_number] = images[i//5*6:i//5*6+6] + positions_X[i:i+5] + positions_Y[i:i+5]
        trial_number += 1

    # I want the dictionary in a pandas dataframe
    columns = ["st{}".format(i) for i in [5, 1, 2, 3, 4]] + ["searchimage"] + ["X{}".format(i) for i in [5, 1, 2, 3, 4]] + ["Y{}".format(i) for i in [5, 1, 2, 3, 4]]
    df = pd.DataFrame.from_dict(trials, orient='index', columns=columns)
    # Reorder columns
    df = df[["st{}".format(i+1) for i in range(5)] + ["searchimage"] + ["X{}".format(i + 1) for i in range(5)] + ["Y{}".format(i + 1) for i in range(5)]]

    return df


for subject_code in exp_info.subjects_ids:
    bh_path_subj = bh_path + subject_code + '/'
    for file in os.listdir(bh_path_subj):
        if file[-4:] == ".log":
            df = parse_images(bh_path_subj + file)
            if len(df) != 210:
                print('WARNING')
            df.to_csv(bh_path_subj + 'ms_items_pos.csv', index=False)
