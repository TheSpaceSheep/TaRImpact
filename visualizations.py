import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from data_loading import load_data
from  matplotlib.ticker import FuncFormatter
from utils import *

save_folder = "/home/noe/Documents/StageLPI/data_processing/plots/"

participants, demographics, tses, workshop_participation, workshop_info = load_data()


def gender_distribution(save=False, show=False):
    teachers = get_teachers()
    gender_df = teachers[["user_id", "Genre"]].drop_duplicates(subset='user_id', keep='first')

    n_profs = len(gender_df)
    ax = sns.countplot(x="Genre", data=gender_df)
    ax.bar_label(ax.containers[0], fmt=lambda x: f'{int(x)} ({round(100*x/n_profs)}%)')
    plt.legend(title=f'n = {n_profs}')
    if save: plt.savefig(save_folder + "gender_distr.png")
    if show: plt.show()


def yexp_teach_distribution(save=False, show=False):
    teachers = get_teachers()
    df = teachers[['user_id', 'yexp_teach']].drop_duplicates(subset='user_id', keep='first')
    ax = sns.histplot(x="yexp_teach", data=df,
                      color="#5c82e0", binwidth=1)
    ax.bar_label(ax.containers[0])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    plt.legend(title=f'n = {len(df)}')
    if save: plt.savefig(save_folder + "yexp_teach_distr.png")
    if show: plt.show()


def subscribed_distribution(save=False, show=False):
    teachers = get_teachers()
    teachers["subscribed"] = pd.to_datetime(teachers["subscribed"])
    df = teachers[['user_id', 'subscribed']].drop_duplicates(subset='user_id', keep='first')
    ax = sns.histplot(x="subscribed",
                      data=teachers,
                      color="#5c82e0",
                      binwidth=180)
    ax.bar_label(ax.containers[0])
    plt.xticks(rotation=20, ha='right')
    plt.ylabel("Subscriptions")
    plt.legend(title=f'n = {len(df)}')
    if save: plt.savefig(save_folder + "subscribed_distr.png")
    if show: plt.show()


def level_distribution(save=False, show=False):
    teachers = get_teachers()
    df = teachers[['user_id', 'teaching_level']].drop_duplicates(subset='user_id', keep='first')
    ax = sns.countplot(x="teaching_level",
                      data=teachers,
                      order=["primaire", "secondaire", "supérieur", "autre"])
    ax.bar_label(ax.containers[0], fmt=lambda x: f'{int(x)} ({round(100*x/len(df))}%)')
    plt.legend(title=f'n = {len(df)}')
    if save: plt.savefig(save_folder + "level_distr.png")
    if show: plt.show()


def privpubl_distribution(save=False, show=False):
    teachers = get_teachers()
    teachers['teaching_privpubl'] = teachers['teaching_privpubl'].map(
        lambda x: "Autre" if x == "Organisme de formation pour adultes" else x
    )
    df = teachers[['user_id', 'teaching_privpubl']].drop_duplicates(subset='user_id', keep='first')
    ax = sns.countplot(x="teaching_privpubl",
                       data=df,
                       order=["Public", "Privé", "Public&Privé", "Autre"])
    ax.bar_label(ax.containers[0], fmt=lambda x: f'{int(x)} ({round(100*x/len(df))}%)')
    ax.set(xticklabels=["Public", "Privé", "Public&Privé", "Autre"])
    plt.legend(title=f'n = {len(df)}')
    if save: plt.savefig(save_folder + "privpubl_distr.png")
    if show: plt.show()


def ses_distribution(save=False, show=False):
    values, short_names = get_teaching_ses_values_short_names()
    short_names = [  # the order should be the same as in "values"
        'Particulièrement défavorisé',
        'Particulièrement favorisé',
        'Public Homogène',
        'Public Hétérogène',
    ]
    teachers = get_teachers()
    # todo : difference between Ni particulièrement fav/defav et homogene ni fav/defav ? - done
    df = teachers[['user_id', 'teaching_ses']].drop_duplicates(subset='user_id', keep='first')
    ax = sns.countplot(x="teaching_ses",
                       data=df,
                       order=values,
                      )
    ax.bar_label(ax.containers[0], fmt=lambda x: f'{int(x)} ({round(100*x/len(df))}%)')
    ax.set(xticklabels=short_names)
    plt.xticks(rotation=20, ha='right', wrap=True)
    plt.legend(title=f'n = {len(df)}')
    plt.tight_layout()
    if save: plt.savefig(save_folder + "ses_distr.png")
    if show: plt.show()


def other_role(save=False, show=False):
    teachers = get_teachers()
    df = teachers[['user_id', 'is_trainer_support', 'is_researcher']].drop_duplicates(subset='user_id', keep='first')
    counts = df[['is_trainer_support', 'is_researcher']].apply(pd.Series.value_counts)
    ax = counts.T.plot(kind='bar')
    plt.xticks(rotation=0, ha='center')
    ax.bar_label(ax.containers[0], fmt=lambda x: f'{int(x)} ({round(100*x/len(df))}%)')
    ax.bar_label(ax.containers[1], fmt=lambda x: f'{int(x)} ({round(100*x/len(df))}%)')
    if save: plt.savefig(save_folder + "other_role.png")
    if show: plt.show()


def workshop_types(save=False, show=False):
    df = workshop_info
    ax = sns.countplot(x="workshop_mode", data=df)
    ax.bar_label(ax.containers[0], fmt=lambda x: f'{int(x)} ({round(100*x/len(df))}%)')
    ax.set(xticklabels=["En personne", "En ligne", "Hybride"])
    plt.legend(title=f'n = {len(df)}')
    if save: plt.savefig(save_folder + "workshop_types.png")
    if show: plt.show()


def workshop_dates(save=False, show=False):
    df = workshop_info
    df = df[['wk_id', 'workshop_date']]
    ax = sns.histplot(x="workshop_date",
                      data=df,
                      color="#5c82e0",
                      binwidth=180)
    ax.bar_label(ax.containers[0])
    plt.xticks(rotation=10, ha='right')
    plt.ylabel("Number of workshops")
    plt.legend(title=f'n = {len(df)}')
    plt.tight_layout()
    if save: plt.savefig(save_folder + "workshop_dates.png")
    if show: plt.show()


def workshop_durations(save=False, show=False):
    df = workshop_info
    df = df[['wk_id', 'workshop_length']]
    ax = sns.histplot(x="workshop_length",
                      data=df,
                      color="#5c82e0",
                      binwidth=30)
    ax.bar_label(ax.containers[0])
    plt.xticks([1, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], rotation=0, ha='right')
    plt.ylabel("Number of workshops")
    plt.xlabel("Workshop duration (in minutes)")
    plt.legend(title=f'n = {len(df)}')
    if save: plt.savefig(save_folder + "workshop_durations.png")
    if show: plt.show()


def workshops_with_k_participants(save=False, show=False):
    df = pd.merge(workshop_info["wk_id"],
                  workshop_participation[["wk_id", "user_id"]],
                  on="wk_id",
                  how="left").groupby("wk_id").count()
    sns.histplot(x="user_id",
                      data=df,
                      binwidth=1)
    plt.legend(title=f'n = {len(df)}')
    plt.xlabel("Number of participants k")
    plt.ylabel("Number of workshops with k participants")
    if save: plt.savefig(save_folder + "workshop_with_k_participants.png")
    if show: plt.show()


def participants_with_k_workshops(save=False, show=False):
    df = pd.merge(participants["user_id"],
                  workshop_participation[["wk_id", "user_id"]],
                  on="user_id",
                  how="left").groupby("user_id").count()
    sns.histplot(x="wk_id",
                      data=df,
                      binwidth=1)
    plt.legend(title=f'n = {len(df)}')
    plt.xlabel("Number of workshops k")
    plt.ylabel("Number of participants who attended k workshops")
    if save: plt.savefig(save_folder + "participants_attended_k_workshops.png")
    if show: plt.show()


def gender_by_ses(save, show):
    values, short_names = get_teaching_ses_values_short_names()
    teachers = get_teachers()
    teachers = fix_teaching_ses(teachers)
    df = teachers[['user_id', 'teaching_ses', 'Genre']].drop_duplicates(subset='user_id', keep='first')

    df_grouped = df.groupby('teaching_ses')['Genre'].value_counts(normalize=True).unstack('Genre')
    df_grouped = df_grouped[['Not asked', 'A/B/NR', 'F', 'M']]
    ax = df_grouped.plot(kind='bar', stacked=True, )

    for c in ax.containers:
        #if the segment is small or 0, customize the labels
        labels = [f'{int(v.get_height()*100)}%' if v.get_height() >= 0.1 else '' for v in c]

        # remove the labels parameter if it's not needed for customized labels
        ax.bar_label(c, labels=labels, label_type='center')

    ax.set(xticklabels=short_names)
    plt.xticks(rotation=20, ha='right', wrap=True)
    plt.tight_layout()
    if save: plt.savefig(save_folder + "gender_by_ses.png")
    if show: plt.show()


def gender_by_level(save, show):
    teachers = get_teachers()
    df = teachers[['user_id', 'teaching_level', 'Genre']].drop_duplicates(subset='user_id', keep='first')

    df_grouped = df.groupby('teaching_level')['Genre'].value_counts(normalize=True).unstack('Genre')
    df_grouped = df_grouped[['Not asked', 'A/B/NR', 'F', 'M']]
    ax = df_grouped.plot(kind='bar', stacked=True, )

    for c in ax.containers:
        #if the segment is small or 0, customize the labels
        labels = [f'{int(v.get_height()*100)}%' if v.get_height() >= 0.1 else '' for v in c]

        # remove the labels parameter if it's not needed for customized labels
        ax.bar_label(c, labels=labels, label_type='center')

    plt.xticks(rotation=20, ha='right', wrap=True)
    plt.tight_layout()
    if save: plt.savefig(save_folder + "gender_by_level.png")
    if show: plt.show()


def tses_baseline(save=False, show=False):
    #TODO: Remove participants who filled TSES form after participating in a workshop ! - done
    #TODO: same in the other tses functions
    #TODO: remove outlier(s)
    teachers = get_teachers()
    df = pd.merge(teachers['user_id'],
                  tses.sort_values("Timestamp").drop_duplicates(subset='user_id', keep='first'),
                  how='inner',
                  on='user_id',
                 ).drop_duplicates(subset='user_id', keep='first').copy()

    # remove users who filled their first TSES survey after participating in a workshop
    not_baseline = pd.merge(
        pd.merge(df, workshop_participation, on='user_id'),
        workshop_info,
        on='wk_id'
    ).sort_values("workshop_date").drop_duplicates(subset='user_id', keep='first')
    not_baseline = not_baseline[not_baseline['Timestamp'].dt.date > not_baseline['workshop_date']]

    df = df[~df['user_id'].isin(not_baseline['user_id'])]

    df["mean"] = df[tses_cols].mean(axis=1)

    sns.histplot(
        x="mean",
        data=df,
        binwidth=0.02
    )
    plt.xlim([0.1, 1.1])
    plt.xlabel("Baseline TSES")
    plt.ylabel("Teachers")
    plt.tight_layout()
    plt.legend(title=f'n = {len(df)}')
    if save: plt.savefig(save_folder + "tses_baseline_distr.png")
    if show: plt.show()


def tses_baseline_subscales(save=False, show=False):
    teachers = get_teachers()
    df = pd.merge(teachers['user_id'],
                  tses,
                  how='inner',
                  on='user_id',
                 ).drop_duplicates(subset='user_id', keep='first').copy()

    df["mean"] = df[tses_cols].mean(axis=1)

    student_engagement = [col for col in df.columns if "engage" in col]
    instructional_strategies = [col for col in df.columns if "strat" in col]
    classroom_management = [col for col in df.columns if "mgmt" in col]

    df["mean_engage"] = df[student_engagement].mean(axis=1)
    df["mean_strat"] = df[instructional_strategies].mean(axis=1)
    df["mean_mgmt"] = df[classroom_management].mean(axis=1)

    _, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.xlim([0.1, 1.1])
    sns.histplot(x="mean_engage", data=df, binwidth=0.05, label="Student Engagement", ax=axs[0, 0], color='blue')
    sns.histplot(x="mean_strat", data=df, binwidth=0.05, label="Instructional Strategies", ax=axs[0, 1], color='crimson')
    sns.histplot(x="mean_mgmt", data=df, binwidth=0.05, label="Classroom Management", ax=axs[1, 0], color='green')

    sns.kdeplot(x="mean_engage", data=df, label="Student Engagement", ax=axs[1, 1], color='blue')
    sns.kdeplot(x="mean_strat", data=df, label="Instructional Strategies", ax=axs[1, 1], color='crimson')
    sns.kdeplot(x="mean_mgmt", data=df, label="Classroom Management", ax=axs[1, 1], color='green')
    mmean_engage = df["mean_engage"].mean()
    mmean_strat = df["mean_strat"].mean()
    mmean_mgmt = df["mean_mgmt"].mean()
    plt.plot([mmean_engage, mmean_engage], [0, np.interp(mmean_engage, axs[1, 1].lines[0].get_xdata(), axs[1, 1].lines[0].get_ydata())], color='blue')
    plt.plot([mmean_strat, mmean_strat], [0, np.interp(mmean_strat, axs[1, 1].lines[1].get_xdata(), axs[1, 1].lines[1].get_ydata())], color='crimson')
    plt.plot([mmean_mgmt, mmean_mgmt], [0, np.interp(mmean_mgmt, axs[1, 1].lines[2].get_xdata(), axs[1, 1].lines[2].get_ydata())], color='green')
    plt.xlabel("Baseline TSES")
    for ax in axs.flatten(): ax.set_ylabel("Teachers")
    plt.ylabel("Kernel Density Estimate")
    plt.tight_layout()
    plt.legend(title=f'n = {len(df)}')
    if save: plt.savefig(save_folder + "tses_baseline_subscales.png")
    if show: plt.show()


def tses_baseline_by(subpop, save=False, show=False):
    values, short_names = get_teaching_ses_values_short_names()
    teachers = get_teachers()
    df = pd.merge(teachers,
                  tses,
                  how='inner',
                  on='user_id',
                 ).drop_duplicates(subset='user_id', keep='first').copy()

    df["mean"] = df[tses_cols].mean(axis=1)

    sns.set_style("whitegrid")
    ax = sns.boxplot(
        x=subpop,
        y="mean",
        data=df,
        palette='cool'
    )

    if subpop == "teaching_ses":
        ax.set(xticklabels=short_names)
        plt.xticks(rotation=20, ha='right', wrap=True)
    elif subpop == "yexp_teach":
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
        ax.figure.set_figwidth(15)
    elif "is_" in subpop:
        ax.figure.set_figwidth(2)

    ax.set_ylabel("Mean TSE")
    plt.tight_layout()
    if save: plt.savefig(save_folder + f"tses_baseline_by_{subpop}.png")
    if show: plt.show()


def tses_by_date(save=False, show=False):
    df = tses

    sns.scatterplot(
        x='Timestamp',
        y='mean',
        data=df,
        color='blue',
    )
    plt.legend()
    plt.xlabel("Survey date")
    plt.ylabel("TSE")
    plt.tight_layout()

    if save: plt.savefig(save_folder + f"tses_by_date.png")
    if show: plt.show()


def tses_by_date_baseline_final(save=False, show=False):
    df = get_processed_tses()

    sns.scatterplot(
        x='Timestamp_baseline',
        y='baseline_mean',
        data=df,
        color='blue',
        label='Baseline TSE',
    )
    sns.scatterplot(
        x='Timestamp_final',
        y='final_mean',
        data=df,
        color='red',
        label='Final TSE',
    )
    plt.legend()
    plt.xlabel("Survey date")
    plt.ylabel("TSE")
    plt.tight_layout()

    if save: plt.savefig(save_folder + f"tses_by_date_baseline_final.png")
    if show: plt.show()


def tses_final_by_baseline(save=False, show=False):
    df = get_processed_tses()

    sns.scatterplot(
        x='Timestamp_baseline',
        y='baseline_mean',
        data=df
    )

    if save: plt.savefig(save_folder + f"tses_final_by_baseline.png")
    if show: plt.show()


def nwks_distr_se(save=False, show=False):
    df = get_processed_tses()
    sns.histplot(data=df, x='nwks', binwidth=1)

    plt.legend(title=f'n = {len(df)}')
    plt.xlabel("Number of workshops k")
    plt.ylabel("Number of participants who attended k workshops")

    if save: plt.savefig(save_folder + f"nwks_distr_se.png")
    if show: plt.show()


def attendance_duration_distr(save=False, show=False):
    participation = pd.merge(participants,
                             workshop_participation,
                             on='user_id', how='left')
    df = pd.merge(participation,
                  workshop_info[['wk_id', 'workshop_length']],
                  on='wk_id',
                  how='left')

    df['attendance_duration'] = df.groupby('user_id')['workshop_length'].sum()
    df = df.drop_duplicates(subset='user_id', keep='first')

    ax = sns.histplot(x="attendance_duration",
                      data=df,
                      color="#5c82e0",
                      binwidth=60)
    ax.figure.set_figwidth(15)
    plt.legend(title=f'n = {len(df)}')
    plt.xticks([0, 60, 120, 180] + list(range(300, 3000, 300)), rotation=0, ha='right')
    plt.tight_layout()
    if save: plt.savefig(save_folder + "attendance_duration_distr.png")
    if show: plt.show()


def nwks_by_duration(save=False, show=False):
    participation = pd.merge(participants,
                             workshop_participation,
                             on='user_id', how='left')
    df = pd.merge(participation,
                  workshop_info[['wk_id', 'workshop_length']],
                  on='wk_id',
                  how='left')

    df['attendance_duration'] = df.groupby('user_id')['workshop_length'].sum()
    df['nwks'] = df.groupby('user_id')['wk_id'].count()
    df = df.drop_duplicates(subset='user_id', keep='first')
    print(len(df))

    ax = sns.scatterplot(
        x='nwks',
        y='attendance_duration',
        data=df
    )
    if save: plt.savefig(save_folder + "attendance_duration_distr.png")
    if show: plt.show()


if __name__=="__main__":
    save = True
    show = True
    ### DEMOGRAPHICS
    #gender_distribution(save, show)
    #yexp_teach_distribution(save, show)
    #subscribed_distribution(save, show)
    #level_distribution(save, show)
    #privpubl_distribution(save, show)
    #ses_distribution(save, show)
    #other_role(save, show)
    #workshop_types(save, show)
    #workshop_dates(save, show)
    #workshop_durations(save, show)
    #workshops_with_k_participants(save, show)
    #participants_with_k_workshops(save, show)

    ### CROSSED DEMOGRAPHICS
    #gender_by_ses(save, show)
    #gender_by_level(save, show)

    ### TSES
    #tses_baseline(save, show)
    #tses_baseline_subscales(save, show)
    #tses_baseline_by("teaching_ses", save, show)
    #tses_baseline_by("teaching_level", save, show)
    #tses_baseline_by("Genre", save, show)
    #tses_baseline_by("yexp_teach", save, show)
    #tses_baseline_by("is_trainer_support", save, show)
    #tses_baseline_by("is_researcher", save, show)
    #tses_by_date(save, show)
    #tses_final_by_baseline(save, show)
    #nwks_distr_se(save, show)  # se for "sous-échantillon"

    attendance_duration_distr(save, show)
    nwks_by_duration(save, show)