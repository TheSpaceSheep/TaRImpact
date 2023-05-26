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

import numpy as np
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from data_loading import load_data
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns

# # Data loading

df = get_processed_tses()

# + [markdown] heading_collapsed=true
# # Reproducing Pagnotta et al.

# + [markdown] hidden=true
# ## Basic linear regression

# + [markdown] heading_collapsed=true hidden=true
# ### Model comparison

# + [markdown] hidden=true
# We compare four models to the baseline model M1 (that predicts final TSE using
# only covariates). Using the likelihood ratio test, we are able to determine
# what model explains the outcome the most while staying parcimonious.

# + hidden=true
models = [
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre',                                      # M1

    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks',                               # M2

    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:baseline_overall_tses',  # M3
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:yexp_teach',             # M4
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:Genre',                  # M5
]

# + hidden=true
# fitting models
results = {}
for model in models:
    y, X = dmatrices(model, data=df, return_type='dataframe')
    mod = sm.OLS(y, X)
    res = mod.fit()
    results[model] = res

# + [markdown] hidden=true
# Performing likelihood ratio tests : higher lr_stat means that adding new
# variables yields an improvement compared to M1

# + hidden=true
for model in models[1:]:
    lr_stat, p_value, _ = results[model].compare_lr_test(results[models[0]])
    print(model)
    print('LR: ', lr_stat, 'pval: ', p_value, '\n')

# + [markdown] hidden=true
# The best model seems to be M5

# + [markdown] heading_collapsed=true hidden=true
# ### Fitting M5 with the 3 TSES subscales

# + hidden=true
models = [
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:Genre',
    'final_mgmt ~ baseline_mgmt + yexp_teach + Genre + nwks + nwks:Genre',
    'final_engage ~ baseline_engage + yexp_teach + Genre + nwks + nwks:Genre',
    'final_strat ~ baseline_strat + yexp_teach + Genre + nwks + nwks:Genre',
]

# + hidden=true
for model in models:
    y, X = dmatrices(model, data=df, return_type='dataframe')
    mod = sm.OLS(y, X)
    res = mod.fit()
    print(model)
    print(f"R squared: {round(res.rsquared, 3)}")
    c_int = list((res.conf_int().loc['nwks']))
    c_int_inter = list((res.conf_int().loc['nwks:Genre[T.M]']))
    pval = res.pvalues.loc['nwks']
    pval_inter = res.pvalues.loc['nwks:Genre[T.M]']
    print(f"nwks : {round(res.params.loc['nwks'], 4)} [{round(c_int[0], 4)}, {round(c_int[1], 4)}], pval: {round(pval, 3)}")
    print(f"nwks:Genre[T.M] : {round(res.params.loc['nwks:Genre[T.M]'], 4)} [{round(c_int_inter[0], 4)}, {round(c_int_inter[1], 4)}], pval: {round(pval_inter, 3)}\n")

# + [markdown] hidden=true
# In general, we find no effect of nwks on final TSE
# Except for student engagement, where we find a positive
# effect of nwks on final TSE only for male teachers
# "Male teachers might feel more able to engage their students
# after participating in the Teachers as Researchers program"

# + [markdown] hidden=true
# ## Mixed Linear Models

# + [markdown] hidden=true
# When doing basic linear regression, we might overlook random effects of each individuals
# A teacher's self-efficacy might not react the same way to exposure to the program, and
# we should account for that using random intercepts and slopes

# + [markdown] hidden=true
# We redo the previous analysis (including model comparison) with mixed linear models
# and report the results.

# + [markdown] heading_collapsed=true hidden=true
# ### Model Comparison

# + hidden=true
models = [
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre',                                      # M1

    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks',                               # M2

    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:baseline_overall_tses',  # M3
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:yexp_teach',             # M4
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:Genre',                  # M5
]

# + hidden=true
# fitting models
results = {}
for model in models:
    mod = smf.mixedlm(model, df, groups=df["user_id"], re_formula="~nwks")
    res = mod.fit(method=['lbfgs'], reml=False)
    results[model] = res

# + [markdown] hidden=true
# We use AIC to compare our models. A higher AIC means the model better explains the data

# + hidden=true
for model in models:
    aic = results[model].aic
    print(model)
    print("aic:", round(aic, 2))

# + [markdown] hidden=true
# The lowest values for AIC are for M1 (only covariates) and M2 (no interaction effects).
# For models with interaction effects, M5 is once again the preferred model.

# + [markdown] heading_collapsed=true hidden=true
# ### Fitting M5 with the 3 TSES subscales

# + hidden=true
models = [
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:Genre',
    'final_mgmt ~ baseline_mgmt + yexp_teach + Genre + nwks + nwks:Genre',
    'final_engage ~ baseline_engage + yexp_teach + Genre + nwks + nwks:Genre',
    'final_strat ~ baseline_strat + yexp_teach + Genre + nwks + nwks:Genre',
]

# + hidden=true
for model in models:
    mod = smf.mixedlm(model, df, groups=df["user_id"], re_formula="~nwks")
    res = mod.fit(method=['lbfgs'])
    print(f"\nModèle : {model}")
    c_int = list((res.conf_int().loc['nwks']))
    pval = res.pvalues.loc['nwks']
    #print(f"nwks : {round(res.params.loc['nwks'], 4)} [{round(c_int[0], 4)}, {round(c_int[1], 4)}], pval: {round(pval, 3)}\n")
    print(res.summary())

# + [markdown] hidden=true
# After adjusting for random effects, we find that there is a slight effect
# on instructional strategies. We also again find an interaction effect for student
# engagement : nwks has an effect on male participants.


# + [markdown] heading_collapsed=true
# # Adding other covariates in the models

# + [markdown] hidden=true
# We add the new measured covariates to our models :
# teaching_ses (socio-economic status of the participant's school),
# is_researcher (is the participant also a researcher),
# teaching_privpubl (are they teaching at a public or private school)
# is_trainer_support (are they also training or supporting other teachers)

# + [markdown] hidden=true
# Notes :
# - We find an effect of "is_researcher" on increase in self-efficacy but it
# should be interpreted with caution, as there are not many teachers who are also researchers
# (82 are NOT researchers
#  13 are researchers)
# - is_trainer_support predicts baseline TSE, but does it have an effect
# on final TSE, when baseline TSE is controlled for ?

# + [markdown] hidden=true
# ## Effect of adding the new covariates to the model

# + hidden=true
models = [
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre',
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + is_trainer_support',
]

# + hidden=true
# fitting the models
results = {}
for model in models:
    y, X = dmatrices(model, data=df, return_type='dataframe')
    mod = sm.OLS(y, X)
    res = mod.fit()
    results[model] = res

# + [markdown] hidden=true
# model comparison : higher lr_stat means that adding new variables
# yields an improvement compared to M1

# + hidden=true
for model in models[1:]:
    lr_stat, p_value, _ = results[model].compare_lr_test(results[models[0]])
    print(model)
    print('LR: ', lr_stat, 'pval: ', p_value, '\n')


# + [markdown] hidden=true
# Adding the covariates yields a significant improvement

# + [markdown] hidden=true
# We now repeat previous analysis with all covariables

# + [markdown] hidden=true
# ## Basic linear regression

# + hidden=true
covariables = 'baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + is_trainer_support'
models = [
    f'final_overall_tses ~ {covariables}',                                     # M1
    f'final_overall_tses ~ {covariables} + nwks',                              # M2
    f'final_overall_tses ~ {covariables} + nwks + nwks:baseline_overall_tses', # M3
    f'final_overall_tses ~ {covariables} + nwks + nwks:yexp_teach',            # M4
    f'final_overall_tses ~ {covariables} + nwks + nwks:Genre',                 # M5
]

# + hidden=true
results = {}
for model in models:
    y, X = dmatrices(model, data=df, return_type='dataframe')
    mod = sm.OLS(y, X)
    res = mod.fit()
    results[model] = res

# + [markdown] hidden=true
# ### Model comparison

# + hidden=true
for model in models[1:]:
    lr_stat, p_value, _ = results[model].compare_lr_test(results[models[0]])
    print(model)
    print('LR: ', lr_stat, 'pval: ', p_value, '\n')

# + [markdown] hidden=true
# The best model seems to be M5 again

# + [markdown] hidden=true
# Fitting M5 with the 3 TSES subscales

# + hidden=true
covariables = 'baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl'
models = [
    f'final_overall_tses ~ {covariables} + nwks + nwks:Genre',
    f'final_mgmt ~ {covariables} + nwks + nwks:Genre',
    f'final_engage ~ {covariables} + nwks + nwks:Genre',
    f'final_strat ~ {covariables} + nwks + nwks:Genre',
]
for model in models:
    y, X = dmatrices(model, data=df, return_type='dataframe')
    mod = sm.OLS(y, X)
    res = mod.fit()
    print(model)
    print(f"R squared: {round(res.rsquared, 3)}")
    c_int = list((res.conf_int().loc['nwks']))
    pval = res.pvalues.loc['nwks']
    c_int_inter = list((res.conf_int().loc['nwks:Genre[T.M]']))
    pval = res.pvalues.loc['nwks:Genre[T.M]']
    print(f"nwks : {round(res.params.loc['nwks'], 4)} [{round(c_int[0], 4)}, {round(c_int[1], 4)}], pval: {round(pval, 3)}")
    print(f"nwks : {round(res.params.loc['nwks:Genre[T.M]'], 4)} [{round(c_int_inter[0], 4)}, {round(c_int_inter[1], 4)}], pval: {round(pval_inter, 3)}\n")
    # There might be a weak positive effect on instructional strategies
    # Otherwise, there seems to be no effect

    # note : for student engagement, there is a significant interaction effect with Genre :
    # nwks:Genre[T.M] 0.0075 [0.001, 0.014]  pval: 0.032
    # This means (female=0, male=1) : nkws has an effect on men in this model


# + [markdown] hidden=true
# ## Mixed Linear Models

# + [markdown] hidden=true
# ### Model comparison

# + hidden=true
covariables = 'baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl'
models = [
    f'final_overall_tses ~ {covariables}',                                     # M1
    f'final_overall_tses ~ {covariables} + nwks',                              # M2
    f'final_overall_tses ~ {covariables} + nwks + nwks:baseline_overall_tses', # M3
    f'final_overall_tses ~ {covariables} + nwks + nwks:yexp_teach',            # M4
    f'final_overall_tses ~ {covariables} + nwks + nwks:Genre',                 # M5
]

# + hidden=true
results = {}
for model in models:
    mod = smf.mixedlm(model, df, groups=df["user_id"], re_formula="~nwks")
    res = mod.fit(method=['lbfgs'], reml=False)
    results[model] = res

# + hidden=true
# AIC comparison : the greater difference with the base model indicates the better model
for model in models:
    aic = results[model].aic
    print(model)
    print("aic:", round(aic, 2))

# + [markdown] hidden=true
# M5 has the lower AIC and hence seems to be the better model

# + [markdown] hidden=true
# ### Fitting M5 with the 3 tses subscales

# + hidden=true
covariables = 'baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl'
models = [
    f'final_overall_tses ~ {covariables} + nwks + nwks:Genre',
    f'final_mgmt ~ {covariables} + nwks + nwks:Genre',
    f'final_engage ~ {covariables} + nwks + nwks:Genre',
    f'final_strat ~ {covariables} + nwks + nwks:Genre',
]

# + hidden=true
for model in models:
    mod = smf.mixedlm(model, df, groups=df["user_id"], re_formula="~nwks")
    res = mod.fit(method=['lbfgs'])
    print(f"\nModèle : {model}")
    c_int = list((res.conf_int().loc['nwks']))
    pval = res.pvalues.loc['nwks']
    #print(f"nwks : {round(res.params.loc['nwks'], 4)} [{round(c_int[0], 4)}, {round(c_int[1], 4)}], pval: {round(pval, 3)}\n")
    print(res.summary())

# + [markdown] hidden=true
# We find again a slight effect on the instructional strategies subscale
# and an interaction for student engagement (nwks has an effect on male participants)

# + hidden=true
# Investigating a potential interaction between nwks and is_researcher
covariables = 'baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl'
models = [
    f'final_overall_tses ~ {covariables} + nwks + nwks:is_researcher',
    f'final_mgmt ~ {covariables} + nwks + nwks:is_researcher',
    f'final_engage ~ {covariables} + nwks + nwks:is_researcher',
    f'final_strat ~ {covariables} + nwks + nwks:is_researcher',
]

# + hidden=true
for model in models:
    mod = smf.mixedlm(model, df, groups=df["user_id"], re_formula="~nwks")
    res = mod.fit(method=['lbfgs'])
    print(f"\nModèle : {model}")
    c_int = list((res.conf_int().loc['nwks']))
    pval = res.pvalues.loc['nwks']
    #print(f"nwks : {round(res.params.loc['nwks'], 4)} [{round(c_int[0], 4)}, {round(c_int[1], 4)}], pval: {round(pval, 3)}\n")
    print(res.summary())

# + [markdown] hidden=true
# Result : no interaction effect between nwks and is_researcher
# -


# # Investigating fixed durations between TSES surveys

# We restrain the dataset to a fixed duration (e.g. 6 months, +/-1 month) between 2 TSES surveys and look if the
# program had an effect TSE over that period. it has been shown that accumulating too many hours of PD can have a
# negative effect on teachers. By choosing a fixed duration, we hope to avoid such potential negative effects.

for i in range(3, 8):
    print(f"Span : {i} months")
    df = get_processed_tses_for_span(i)
    df = df.applymap(lambda x: x[:18] if isinstance(x, str) else x)

    covariables = 'baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl'
    models = [
        f'final_overall_tses ~ {covariables}',                                     # M1
        f'final_overall_tses ~ {covariables} + nwks',                              # M2
        f'final_overall_tses ~ {covariables} + nwks + nwks:baseline_overall_tses', # M3
        f'final_overall_tses ~ {covariables} + nwks + nwks:yexp_teach',            # M4
        f'final_overall_tses ~ {covariables} + nwks + nwks:Genre',                 # M5
    ]

    results = {}
    for model in models:
        mod = smf.mixedlm(model, df, groups=df["user_id"], re_formula="~nwks")
        res = mod.fit(method=['lbfgs'], reml=False)
        results[model] = res
        print(res.summary())


# We find again no effect, except for spans of 4 and 5 months. For these spans, we find that nwks has a significant
# positive effect on final TSE. There seems to be an interaction with Genre and, to a lesser extent, with baseline_tse.
# However, except for these 2 occurences, there are no effects to report for all other cases. Since we are dealing with
# small sample sizes in this last analysis, we should proceed with caution.

