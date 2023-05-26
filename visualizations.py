# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Imports

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from data_loading import load_data
from  matplotlib.ticker import FuncFormatter
from utils import *


# # Data Loading

save_folder = "/home/noe/Documents/StageLPI/data_processing/plots/"
participants, demographics, tses, workshop_participation, workshop_info = load_data()


# + [markdown] heading_collapsed=true
# # Plotting functions definitions

# + hidden=true
def gender_distribution(save=False, show=False):
    teachers = get_teachers()
    gender_df = teachers[["user_id", "Genre"]].drop_duplicates(subset='user_id', keep='first')

    n_profs = len(gender_df)
    ax = sns.countplot(x="Genre", data=gender_df)
    ax.bar_label(ax.containers[0], fmt=lambda x: f'{int(x)} ({round(100*x/n_profs)}%)')
    plt.legend(title=f'n = {n_profs}')
    if save: plt.savefig(save_folder + "gender_distr.png")
    if show: plt.show()


# + hidden=true
def yexp_teach_distribution(save=False, show=False):
    teachers = get_teachers()
    df = teachers[['user_id', 'yexp_teach']].drop_duplicates(subset='user_id', keep='first')
    ax = sns.countplot(x="yexp_teach", data=df,
                      color="#5c82e0")#, binwidth=2)
    #ax.bar_label(ax.containers[0])
    #ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    plt.legend(title=f'n = {len(df)}')
    if save: plt.savefig(save_folder + "yexp_teach_distr.png")
    if show: plt.show()


# + hidden=true
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


# + hidden=true
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


# + hidden=true
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


# + hidden=true
def ses_distribution(save=False, show=False):
    teachers = get_teachers()
    values, short_names = get_teaching_ses_values_short_names()
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


# + hidden=true
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


# + hidden=true
def workshop_types(save=False, show=False):
    df = workshop_info
    ax = sns.countplot(x="workshop_mode", data=df)
    ax.bar_label(ax.containers[0], fmt=lambda x: f'{int(x)} ({round(100*x/len(df))}%)')
    ax.set(xticklabels=["En personne", "En ligne", "Hybride"])
    plt.legend(title=f'n = {len(df)}')
    if save: plt.savefig(save_folder + "workshop_types.png")
    if show: plt.show()


# + hidden=true
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


# + hidden=true
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


# + hidden=true
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


# + hidden=true
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


# + hidden=true
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


# + hidden=true
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


# + hidden=true
def tses_baseline(save=False, show=False):
    df = get_processed_tses()

    sns.histplot(
        x="baseline_overall_tses",
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


# + hidden=true
def tses_baseline_subscales(save=False, show=False):
    teachers = get_teachers()
    df = pd.merge(teachers['user_id'],
                  tses,
                  how='inner',
                  on='user_id',
                 ).drop_duplicates(subset='user_id', keep='first').copy()

    tses_cols = [col for col in df.columns if 'tses' in col]
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


# + hidden=true
def tses_baseline_by(subpop, save=False, show=False):
    values, short_names = get_teaching_ses_values_short_names()
    teachers = get_teachers()
    df = pd.merge(teachers,
                  tses,
                  how='inner',
                  on='user_id',
                 ).drop_duplicates(subset='user_id', keep='first').copy()

    tses_cols = [col for col in df.columns if 'tses' in col]
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


# + hidden=true
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


# + hidden=true
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


# + hidden=true
def tses_final_by_baseline(save=False, show=False):
    df = get_processed_tses()

    sns.scatterplot(
        x='baseline_overall_tses',
        y='final_overall_tses',
        hue='Genre',
        data=df
    )

    if save: plt.savefig(save_folder + f"tses_final_by_baseline.png")
    if show: plt.show()


# + hidden=true
def nwks_distr_se(save=False, show=False):
    df = get_processed_tses()
    sns.histplot(data=df, x='nwks', binwidth=1)

    plt.legend(title=f'n = {len(df)}')
    plt.xlabel("Number of workshops k")
    plt.ylabel("Number of participants who attended k workshops")

    if save: plt.savefig(save_folder + f"nwks_distr_se.png")
    if show: plt.show()


# + hidden=true
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


# + hidden=true
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
    plt.plot([0, 18], [0, 18*3*60], label='3h', linewidth=1)
    plt.plot([0, 18], [0, 18*2.5*60], label='2h30', linewidth=1)
    plt.plot([0, 18], [0, 18*2*60], label='2h', linewidth=1)
    plt.plot([0, 18], [0, 18*1.5*60], label='1h30', linewidth=1)
    plt.plot([0, 18], [0, 18*1*60], label='1h', linewidth=1)
    plt.legend()
    if save: plt.savefig(save_folder + "attendance_duration_distr.png")
    if show: plt.show()


# + hidden=true
def nb_teachers_vs_tses_spans(save=False, show=False):
    """
    Displays, for each 'span' (period between baseline survey and another survey),
    the number of teachers for whom we have data over that span
    note : the total number is greater than the number of teachers, because we may
    have data fitting several spans for one teacher
    """
    teachers = get_teachers().drop_duplicates(subset='user_id', keep='first')
    df = pd.merge(teachers[[c for c in teachers.columns if c != 'Timestamp']], tses, on='user_id', how='inner')
    df['Timestamp_baseline'] = pd.merge(df, df.groupby('user_id')['Timestamp'].min(), on='user_id')['Timestamp_y']
    df['span'] = (df['Timestamp'] - df['Timestamp_baseline']).dt.days
    df = df[df['span'] > 0]  # only keep teachers who filled >=2 surveys
    avg_days_by_month = 30.4167
    df['span_in_months'] = df['span'] / avg_days_by_month
    sns.histplot(x='span_in_months', data=df, binwidth=1)
    plt.legend(title=f"number of points={len(df)}\nnumber of individual teachers = {len(df.drop_duplicates(subset='user_id', keep='first'))}\n span : time between baseline \nsurvey and some later survey")
    plt.ylabel('Number of teachers')
    plt.xlabel('Span (time between baseline survey and some later survey, in months)')
    if save: plt.savefig(save_folder + "nb_teachers_vs_tses_spans.png")
    if show: plt.show()


# + hidden=true
def total_wkshours_distr(save=False, show=False):
    teachers = get_teachers().drop_duplicates(subset='user_id', keep='first')
    df = pd.merge(teachers, workshop_participation, on='user_id', how='left')
    df = pd.merge(df, workshop_info, on='wk_id', how='left')
    df['total_wkshours'] = pd.merge(df, df.groupby('user_id')['workshop_length'].sum(), on='user_id', how='right')['workshop_length_y']
    df['total_wkshours'] = df['total_wkshours'] / 60
    df = df.drop_duplicates(subset='user_id', keep='first')

    sns.histplot(x='total_wkshours', data=df, binwidth=1)
    plt.legend(title=f"n={len(df)}")
    plt.ylabel('Number of teachers')
    plt.xlabel('Total time spent in workshops (in hours)')
    if save: plt.savefig(save_folder + "total_wkshours_distr.png")
    if show: plt.show()


# + hidden=true
def wkshours_vs_intervention_spans(save=False, show=False):
    """
    Displays, for each 'span' (period between a teacher's first and last workshop),
    the number of hours spent in workshop for that teacher
    """
    teachers = get_teachers().drop_duplicates(subset='user_id', keep='first')
    df = pd.merge(teachers, workshop_participation, on='user_id', how='left')
    df = pd.merge(df, workshop_info, on='wk_id', how='left')

    # compute span
    df['first_wk_date'] = pd.merge(df, df.groupby('user_id')['workshop_date'].min(), on='user_id', how='inner')['workshop_date_y']
    df['last_wk_date'] = pd.merge(df, df.groupby('user_id')['workshop_date'].max(), on='user_id', how='inner')['workshop_date_y']
    df['span'] = (df['last_wk_date'] - df['first_wk_date']).dt.days
    avg_days_by_month = 30.4167
    df['span_in_months'] = (df['span'] / avg_days_by_month).replace(np.nan, 0.)

    # compute total workshop hours
    # merge 'right', to discard users who attended less than 2 workshops
    df['total_wkshours'] = pd.merge(df, df.groupby('user_id')['workshop_length'].sum(), on='user_id', how='left')['workshop_length_y']
    df['total_wkshours'] = df['total_wkshours'] / 60

    # compute number of workshops
    df['nwks'] = pd.merge(df, df.groupby('user_id')['wk_id'].count(), on='user_id', how='left')['wk_id_y']

    df = df[df['span'] > 0]  # removes teachers who attended 0 or 1 workshops
    df = df.drop_duplicates(subset='user_id', keep='first')

    #sns.stripplot(x='span_in_months', y='total_wkshours', jitter=0.2, data=df)
    sns.scatterplot(x=jitter(df['span_in_months'], 0.3), y=jitter(df['total_wkshours'], 0.3), alpha=0.9, hue=df['nwks'])
    plt.legend(title=f"n={len(df)}\nColor for number\nof workshops :")
    plt.ylabel('Total time spent in workshops (hours)')
    plt.xlabel('Span (time between first and last workshop, in months)')
    if save: plt.savefig(save_folder + "wkshours_vs_intervention_spans.png")
    if show: plt.show()


# + hidden=true
def tses_baseline_by_cov(cov, save, show):
    df = get_processed_tses()
    sns.stripplot(x=cov, y='baseline_overall_tses', data=df, label='baseline_TSE')
    ax = sns.stripplot(x=cov, y='final_overall_tses', data=df, label='final_TSE')
    plt.xticks(rotation=5, ha='right')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend([handles[0]]+[handles[-1]], [labels[0]]+[labels[-1]])
    if save: plt.savefig(save_folder + "tses_baseline_by_ses.png")
    if show: plt.show()


# + hidden=true
def is_researcher_by_nwks(save=False, show=False):
    # problème : pas assez de profs chercheurs pour que la proportion soit fiable.
    # regarder plutôt la proportion d'individus ayant changé is_researcher
    df = get_processed_is_researcher()
    grouped_df = df.groupby('nwks')['is_researcher'].value_counts(normalize=True).unstack()['Oui']
    grouped_df.plot(kind='bar')
    if save: plt.savefig(save_folder + "is_researcher_by_nwks.png")
    if show: plt.show()


# + hidden=true
def changing_researchers(save=False, show=False):
    teachers = get_teachers()
    teachers_teachers = pd.merge(teachers.copy(), teachers.copy(), how='inner', on='user_id')
    teachers_teachers = teachers_teachers[teachers_teachers['Timestamp_x'] < teachers_teachers['Timestamp_y']]
    changing = teachers_teachers[teachers_teachers['is_researcher_x'] != teachers_teachers['is_researcher_y']]
    became_researchers = changing[changing['is_researcher_x'] == 'Non']
    print(became_researchers['user_id'])
    df = pd.merge(became_researchers, get_processed_is_researcher(), on='user_id', how='left').drop_duplicates(subset='user_id', keep='last')
    print(df[['user_id', 'nwks']])


# -

# # Plots

# ## Save Options

save = False
show = True

# ## Demographics

if __name__=="__main__":
    gender_distribution(save, show)
    yexp_teach_distribution(save, show)
    subscribed_distribution(save, show)
    level_distribution(save, show)
    privpubl_distribution(save, show)
    ses_distribution(save, show)
    other_role(save, show)
    workshop_types(save, show)
    workshop_dates(save, show)
    workshop_durations(save, show)
    workshops_with_k_participants(save, show)
    participants_with_k_workshops(save, show)

# + [markdown] heading_collapsed=true
# ## Crossed Demographics

# + hidden=true
if __name__=="__main__":
    gender_by_ses(save, show)
    gender_by_level(save, show)

# + [markdown] heading_collapsed=true
# ## TSES

# + hidden=true
if __name__=="__main__":
    tses_baseline(save, show)
    tses_baseline_subscales(save, show)
    tses_baseline_by("teaching_ses", save, show)
    tses_baseline_by("teaching_level", save, show)
    tses_baseline_by("Genre", save, show)
    tses_baseline_by("yexp_teach", save, show)
    tses_baseline_by("is_trainer_support", save, show)
    tses_baseline_by("is_researcher", save, show)
    tses_by_date(save, show)
    tses_final_by_baseline(save, show)
    for cov in ['Genre', 'yexp_teach', 'teaching_level', 'teaching_privpubl', 'teaching_ses', 'is_researcher', 'nwks']:
        tses_baseline_by_cov(cov, save, show)
# -

# ## Participation

if __name__=="__main__":
    nwks_distr_se(save, show)  # se for "sous-échantillon"
    attendance_duration_distr(save, show)
    nwks_by_duration(save, show)
    nb_teachers_vs_tses_spans(save, show)
    total_wkshours_distr(save, show)
    wkshours_vs_intervention_spans(save, show)

if __name__=="__main__":
    is_researcher_by_nwks(save, show)

if __name__=="__main__":
    changing_researchers(save, show)
