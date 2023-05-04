import pandas as pd
from data_loading import load_data

participants, demographics, tses, workshop_participation = load_data()

print("Nombre total de participants :", len(participants))

# note : these contain duplicates (some teachers filled several demographics surveys)
teachers = pd.merge(participants,
                    demographics[demographics["is_teaching"]=="Oui"],
                    how="inner",
                    on="user_id")
non_teachers = pd.merge(participants,
                        demographics[demographics["is_teaching"]=="Non"],
                        how="inner",
                        on="user_id")

# note : teachers and non_teachers are NOT disjoint
# (some teachers became non-teachers or vice-versa)
print(pd.merge(teachers, non_teachers, on="user_id", how="inner"))

before_2022_09_01 = teachers[
    pd.to_datetime(teachers["subscribed"]) < pd.to_datetime("2022-09-01")
]

# Control group
# Teachers who participated in 0 workshop (wk), and for whom we have self-efficacy data
teachers_nb_wk = pd.merge(teachers, workshop_participation, how="left").groupby("user_id")["wk_id"].count().reset_index()
zero_workshop = teachers_nb_wk[teachers_nb_wk["wk_id"] == 0]
teachers_zero_workshop = pd.merge(teachers, zero_workshop, how='inner', on='user_id')

teachers_tses = pd.merge(teachers, tses, how="left").groupby("user_id")["Timestamp"].count().reset_index()
tse_measured = teachers_tses[teachers_tses["Timestamp"] > 0]
tse_measured_twice = teachers_tses[teachers_tses["Timestamp"] > 1]
teachers_tse_measured = pd.merge(teachers, tse_measured, how='inner', on='user_id')
teachers_tse_measured_twice = pd.merge(teachers, tse_measured_twice, how='inner', on='user_id')

control = pd.merge(teachers_zero_workshop, teachers_tse_measured, how='inner', on='user_id')             # 74 personnes
control_hard = pd.merge(teachers_zero_workshop, teachers_tse_measured_twice, how='inner', on='user_id')  # 14 personnes
# TODO: Remove teachers who were required to participate (intention-to-treat effect)

# TODO: Are there teachers who filled 2 TSES, then participated ?
