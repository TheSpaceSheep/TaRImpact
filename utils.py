from data_loading import load_data
import pandas as pd


def get_teachers():
    participants, demographics, tses, workshop_participation, workshop_info = load_data()

    # keep participants who answered yes to "are you teaching" at least once
    # contains duplicates
    teachers = pd.merge(participants,
                        demographics[demographics["is_teaching"]=="Oui"],
                        how="inner",
                        on="user_id"
               )
    teachers = fix_teaching_ses(teachers)
    return teachers


def fix_teaching_ses(teachers):
    teachers['teaching_ses'] = teachers['teaching_ses'].map(
        lambda x: 'Public homogène, ni particulièrement favorisé ou défavorisé'
        if x == 'Ni particulièrement favorisé ou défavorisé' else x
    )
    return teachers


def get_processed_tses():
    """
    Returns a dataframe of teachers demographics, with the following added columns:
        "baseline_overall_tses":
            the mean of self-efficacy for each teacher, at baseline (surveys that were filled
            before attending any workshop)
        "final_overall_tses":
            the mean of self-efficacy for each teacher, for the latest surveys that were filled
            (only teachers who filled more than one survey)
        #TODO: add subscales (baseline_mgmt, etc.)           [medium] - done
        #TODO: add k_wkshp_tse ({0,1,2}_wkshp_tse etc..)     [hard]
        #TODO: add n_wkshop                                  [easy] -! Only between baseline and last TSES !
    """
    participants, demographics, tses, workshop_participation, workshop_info = load_data()

    teachers = get_teachers()

    # baseline_tse
    baseline_tse = pd.merge(
        teachers[[c for c in teachers.columns if c != 'Timestamp']].drop_duplicates(subset='user_id', keep='first').copy(),
        tses.sort_values("Timestamp").drop_duplicates(subset='user_id', keep='first'),
        how='inner',
        on='user_id',
    )

    # remove users who filled their first TSES survey AFTER participating in a workshop
    not_baseline = pd.merge(
        pd.merge(baseline_tse, workshop_participation, on='user_id'),
        workshop_info,
        on='wk_id',
    ).sort_values("workshop_date").drop_duplicates(subset='user_id', keep='first')
    not_baseline = not_baseline[not_baseline['Timestamp'].dt.date > not_baseline['workshop_date']]

    baseline_tse = baseline_tse[~baseline_tse['user_id'].isin(not_baseline['user_id'])]

    tses_cols = [col for col in baseline_tse.columns if 'tses' in col]
    baseline_tse["baseline_overall_tses"] = baseline_tse[tses_cols].mean(axis=1)

    for subscale in ['mgmt', 'engage', 'strat']:
        cols = [col for col in baseline_tse.columns if subscale in col]
        baseline_tse[f"baseline_{subscale}"] = baseline_tse[cols].mean(axis=1)

    # final_tse
    final_tse = pd.merge(teachers[[col for col in teachers.columns if col != 'Timestamp']],
                  tses.sort_values("Timestamp").drop_duplicates(subset='user_id', keep='last'),
                  how='inner',
                  on='user_id',
                 ).drop_duplicates(subset='user_id', keep='first').copy()

    # remove people who filled only one survey
    only_one_survey = pd.merge(
        teachers[[col for col in teachers.columns if col != 'Timestamp']],
        tses,
        how='inner',
        on='user_id',
    ).groupby('user_id', as_index=False)['Timestamp'].count()
    only_one_survey = only_one_survey[only_one_survey['Timestamp'] == 1]
    final_tse = final_tse[~final_tse['user_id'].isin(only_one_survey['user_id'])]

    tses_cols = [col for col in final_tse.columns if 'tses' in col]
    final_tse["final_overall_tses"] = final_tse[tses_cols].mean(axis=1)

    for subscale in ['mgmt', 'engage', 'strat']:
        cols = [col for col in final_tse.columns if subscale in col]
        final_tse[f"final_{subscale}"] = final_tse[cols].mean(axis=1)


    df = pd.merge(teachers[[c for c in teachers.columns if c != 'Timestamp']],
                  baseline_tse[['user_id',
                                'Timestamp',
                                'baseline_overall_tses',
                                'baseline_mgmt',
                                'baseline_engage',
                                'baseline_strat']],
                  on='user_id',
                  how='inner').drop_duplicates(subset='user_id', keep='first')
    df = df.rename(columns={'Timestamp': 'Timestamp_baseline'})
    df = pd.merge(df, final_tse[['user_id', 'Timestamp', 'final_overall_tses',
                  'final_mgmt', 'final_engage', 'final_strat']], on='user_id', how='inner')
    df = df.rename(columns={'Timestamp': 'Timestamp_final'})
    df = df.drop_duplicates(subset='user_id', keep='first')

    # keep only workshops participations that took place between a user's first and last tses survey
    df_wk = pd.merge(df, workshop_participation, on='user_id', how='left')
    df_wk_info = pd.merge(df_wk, workshop_info, on='wk_id', how='inner')
    valid_wk = df_wk_info[(df_wk_info['Timestamp_baseline'] <= df_wk_info['workshop_date']) & (df_wk_info['workshop_date'] <= df_wk_info['Timestamp_final'])]
    valid_wk_participation = valid_wk[workshop_participation.columns]

    wk_count = pd.merge(df, valid_wk_participation, on='user_id', how='left').groupby('user_id')['wk_id'].count()
    df = pd.merge(df, wk_count, on='user_id', how='left')
    df = df.rename(columns={"wk_id": "nwks"})

    # keep only users who have a gap of min_gap months between their baseline and final tses survey
    min_gap = 3
    df = df[(df['Timestamp_final']-df['Timestamp_baseline']) > pd.to_timedelta(min_gap*30, unit='d')]
    return df


if __name__=='__main__':
    df = get_processed_tses()

