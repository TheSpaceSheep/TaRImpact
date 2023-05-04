import pandas as pd

data = None
#TODO: separate preprocessing from loading

def load_data(data_folder="../Data_2023-04-07/"):
    global data
    if data is not None:
        print("Warning: Skipping data loading (already loaded)")
        return data

    print("Loading data...")
    participants = pd.read_csv(data_folder + "Participants.csv")
    demographics = pd.read_csv(data_folder + "Participant_Demographics.csv")
    tses = pd.read_csv(data_folder + "TSES.csv")
    workshop_participation = pd.read_csv(data_folder + "Workshop_participation.csv")
    workshop_info = pd.read_csv(data_folder + "Workshop_info.csv")

    demographics['Timestamp'] = pd.to_datetime(demographics['Timestamp'])
    tses['Timestamp'] = pd.to_datetime(tses['Timestamp'])
    workshop_info['workshop_date'] = pd.to_datetime(workshop_info['workshop_date'])

    data = participants, demographics, tses, workshop_participation, workshop_info

    for df in data:  # drop index column
        df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    # normalize tses
    print("Normalizing TSES...")
    tses_cols = [col for col in tses.columns if 'tses' in col]
    tses['scale_max'] = (tses['echelle'].copy()=='ech1_5')*5 + (tses['echelle'].copy()=='ech1_9')*9
    for col in [col for col in tses.columns if "tses" in col]:
        tses[col] = (tses[col].copy()-1) / (tses['scale_max']-1)

    # remove outliers in TSES
    # remove points where X = [1.]*12 ?
    print("Removing outliers...")
    tses['mean'] = tses[tses_cols].mean(axis=1)
    print(f"Number of 'all 1.' points : {len(tses[tses['mean']==1.])}")
    tses['z_score'] = (tses['mean'] - tses['mean'].mean()) / tses['mean'].std()
    z_score_threshold = 3
    print(f"Dropping {len(tses[abs(tses['z_score']) > z_score_threshold])} outliers in tses data (of size {len(tses)})")
    tses.drop(tses[abs(tses['z_score']) > z_score_threshold].index, inplace=True)

    return data


if __name__=='__main__':
    participants, demographics, tses, workshop_participation, workshop_info = load_data()

