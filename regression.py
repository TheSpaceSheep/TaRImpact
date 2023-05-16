import numpy as np
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from data_loading import load_data
from utils import *

import matplotlib.pyplot as plt
import seaborn as sns


df = get_processed_tses()

# ______________ Linear regression ________________

# Model comparison, using the likelihood ratio test

models = [
    # Regression models from pagnotta et al.
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre',                                      # M1

    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks',                               # M2

    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:baseline_overall_tses',  # M3
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:yexp_teach',             # M4
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:Genre',                  # M5
]

results = {}  # for later model comparison
for model in models:
    y, X = dmatrices(model, data=df, return_type='dataframe')
    mod = sm.OLS(y, X)
    res = mod.fit()
    results[model] = res


# model comparison : higher lr_stat means that adding new variables yields an improvement
# compared to M1
for model in models[1:]:
    lr_stat, p_value, _ = results[model].compare_lr_test(results[models[0]])
    print(model)
    print('LR: ', lr_stat, 'pval: ', p_value, '\n')
    # Results :

    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks
    # LR:  4.810665142679568 pval:  0.02828412616677737

    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:baseline_overall_tses
    # LR:  5.0312016471714855 pval:  0.08081434245422556

    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:yexp_teach
    # LR:  4.974783376204982 pval:  0.08312650390309326

    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:Genre
    # LR:  5.808773163502508 pval:  0.054782384811339284

    # The best model seems to be M5

# We fit M5 with the 3 TSES subscales
models = [
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:Genre',
    'final_mgmt ~ baseline_mgmt + yexp_teach + Genre + nwks + nwks:Genre',
    'final_engage ~ baseline_engage + yexp_teach + Genre + nwks + nwks:Genre',
    'final_strat ~ baseline_strat + yexp_teach + Genre + nwks + nwks:Genre',
]

for model in models:
    y, X = dmatrices(model, data=df, return_type='dataframe')
    mod = sm.OLS(y, X)
    res = mod.fit()
    print(model)
    print(f"R squared: {round(res.rsquared, 3)}")
    c_int = list((res.conf_int().loc['nwks']))
    pval = res.pvalues.loc['nwks']
    print(f"nwks : {round(res.params.loc['nwks'], 4)} [{round(c_int[0], 4)}, {round(c_int[1], 4)}], pval: {round(pval, 3)}\n")
    # Results :

    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:Genre
    # R squared: 0.338
    # nwks : 0.0006 [-0.0044, 0.0056], pval: 0.811

    # final_mgmt ~ baseline_mgmt + yexp_teach + Genre + nwks + nwks:Genre
    # R squared: 0.387
    # nwks : 0.0001 [-0.0063, 0.0065], pval: 0.97

    # final_engage ~ baseline_engage + yexp_teach + Genre + nwks + nwks:Genre
    # R squared: 0.317
    # nwks : -0.0038 [-0.0095, 0.0019], pval: 0.192

    # final_strat ~ baseline_strat + yexp_teach + Genre + nwks + nwks:Genre
    # R squared: 0.275
    # nwks : 0.0067 [0.0003, 0.013], pval: 0.041

    # There might be a weak positive effect on instructional strategies
    # Otherwise, there seems to be no effect


# ______________ Mixed Linear Models ______________

# When doing basic linear regression, we might overlook random effects of each individuals
# A teacher's self-efficacy might not react the same way to exposure to the program, and
# we should account for that using random intercepts and slopes

# Similarly, a teacher will probably have a tendency to give correlated answers to
# all questions.

# We redo the previous analysis (including model comparison) with mixed linear models
# and report the results.

models = [
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre',                                      # M1

    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks',                               # M2

    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:baseline_overall_tses',  # M3
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:yexp_teach',             # M4
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:Genre',                  # M5
]

results = {}
for model in models:
    mod = smf.mixedlm(model, df, groups=df["user_id"], re_formula="~nwks")
    res = mod.fit(method=['lbfgs'], reml=False)
    results[model] = res


for model in models:
    aic = results[model].aic
    print(model)
    print("aic:", round(aic, 2))
    # Results :

    #final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre
    #aic: -178.59

    #final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks
    #aic: -177.4

    #final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:baseline_overall_tses
    #aic: -175.72

    #final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:yexp_teach
    #aic: -175.78

    #final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:Genre
    #aic: -177.52

    # The best model is M1 according to AIC test.
    # Among the other models, the best one is M5


# We fit M5 with the 3 TSES subscales

models = [
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:Genre',
    'final_mgmt ~ baseline_mgmt + yexp_teach + Genre + nwks + nwks:Genre',
    'final_engage ~ baseline_engage + yexp_teach + Genre + nwks + nwks:Genre',
    'final_strat ~ baseline_strat + yexp_teach + Genre + nwks + nwks:Genre',
]

for model in models:
    mod = smf.mixedlm(model, df, groups=df["user_id"], re_formula="~nwks")
    res = mod.fit(method=['lbfgs'])
    print(f"\nModèle : {model}")
    c_int = list((res.conf_int().loc['nwks']))
    pval = res.pvalues.loc['nwks']
    #print(f"nwks : {round(res.params.loc['nwks'], 4)} [{round(c_int[0], 4)}, {round(c_int[1], 4)}], pval: {round(pval, 3)}\n")
    print(res.summary())

    # Résultats
    # Modèle : final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:Genre
    #                       Coef.  Std.Err.   z    P>|z| [0.025 0.975]
    # Intercept              0.387    0.064  6.089 0.000  0.263  0.512
    # Genre[T.M]            -0.014    0.025 -0.557 0.577 -0.064  0.036
    # baseline_overall_tses  0.487    0.082  5.952 0.000  0.327  0.648
    # yexp_teach            -0.001    0.001 -1.003 0.316 -0.003  0.001
    # nwks                   0.000    0.002  0.101 0.920 -0.004  0.005 <-
    # nwks:Genre[T.M]        0.004    0.003  1.278 0.201 -0.002  0.009 <-
    # Group Var              0.006
    # Group x nwks Cov      -0.000
    # nwks Var               0.000
    #
    # Modèle : final_mgmt ~ baseline_mgmt + yexp_teach + Genre + nwks + nwks:Genre
    #                  Coef.  Std.Err.   z    P>|z| [0.025 0.975]
    # Intercept         0.337    0.069  4.917 0.000  0.203  0.471
    # Genre[T.M]       -0.025    0.031 -0.788 0.431 -0.086  0.037
    # baseline_mgmt     0.573    0.085  6.747 0.000  0.406  0.739
    # yexp_teach       -0.001    0.001 -0.619 0.536 -0.003  0.002
    # nwks              0.000    0.003  0.038 0.969 -0.006  0.006 <-
    # nwks:Genre[T.M]   0.004    0.003  1.255 0.209 -0.002  0.011 <-
    # Group Var         0.009    0.248
    # Group x nwks Cov -0.000    0.002
    # nwks Var          0.000
    #
    # Modèle : final_engage ~ baseline_engage + yexp_teach + Genre + nwks + nwks:Genre
    #                  Coef.  Std.Err.   z    P>|z| [0.025 0.975]
    # Intercept         0.450    0.057  7.959 0.000  0.339  0.561
    # Genre[T.M]       -0.070    0.029 -2.437 0.015 -0.126 -0.014
    # baseline_engage   0.416    0.072  5.750 0.000  0.274  0.558
    # yexp_teach       -0.001    0.001 -0.861 0.389 -0.003  0.001
    # nwks             -0.005    0.002 -2.228 0.026 -0.009 -0.001 <-
    # nwks:Genre[T.M]   0.009    0.003  3.038 0.002  0.003  0.014 <-
    # Group Var         0.009
    # Group x nwks Cov -0.000
    # nwks Var          0.000
    #
    # Modèle : final_strat ~ baseline_strat + yexp_teach + Genre + nwks + nwks:Genre
    #                  Coef.  Std.Err.   z    P>|z| [0.025 0.975]
    # Intercept         0.421    0.072  5.832 0.000  0.280  0.563
    # Genre[T.M]        0.050    0.031  1.639 0.101 -0.010  0.111
    # baseline_strat    0.405    0.091  4.453 0.000  0.227  0.584
    # yexp_teach       -0.001    0.001 -0.912 0.362 -0.004  0.001
    # nwks              0.006    0.003  2.170 0.030  0.001  0.011 <-
    # nwks:Genre[T.M]  -0.003    0.003 -1.035 0.301 -0.008  0.003 <-
    # Group Var         0.011    0.553
    # Group x nwks Cov -0.000    0.006
    # nwks Var          0.000    0.000

    # After adjusting for random effects, we find that there is a slight effect
    # on instructional strategies. We also find an interaction effect for student
    # engagement : nwks has an effect on male participants.


# ____________ Adding the other covariates in the models ____________

models = [
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre',
    'baseline_overall_tses ~ yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + is_trainer_support',
]

# is_trainer_support explique bien la baseline TSE
# mais n'a pas d'effet sur final_tse quand on contrôle par baseline_TSE

results = {}
for model in models:
    y, X = dmatrices(model, data=df, return_type='dataframe')
    mod = sm.OLS(y, X)
    res = mod.fit()
    print(res.summary())
    results[model] = res

    # Results
    # 'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl',
    #                                         coef    std err          t      P>|t|      [0.025      0.975]
    # Intercept                              0.2163      0.117      1.855      0.067      -0.016       0.448
    # Genre[T.M]                             0.0143      0.019      0.758      0.450      -0.023       0.052
    # teaching_ses[T.Particulièrement dé    -0.0088      0.025     -0.357      0.722      -0.058       0.040
    # teaching_ses[T.Particulièrement fa     0.0155      0.034      0.452      0.652      -0.053       0.084
    # teaching_ses[T.Public homogène, ni     0.0207      0.024      0.859      0.393      -0.027       0.069
    # is_researcher[T.Oui]                   0.0750      0.029      2.633      0.010       0.018       0.132  <-
    # teaching_privpubl[T.Privé]             0.0168      0.098      0.172      0.864      -0.177       0.211
    # teaching_privpubl[T.Public]            0.1257      0.094      1.339      0.184      -0.061       0.312
    # teaching_privpubl[T.Public&Privé]      0.0498      0.134      0.371      0.711      -0.217       0.316
    # baseline_overall_tses                  0.5087      0.085      5.993      0.000       0.340       0.677
    # yexp_teach                             0.0003      0.001      0.264      0.792      -0.002       0.002

    # Avec is_trainer_support
    # Intercept                              0.2291      0.078      2.934      0.004       0.074       0.384
    # Genre[T.M]                             0.0121      0.019      0.633      0.528      -0.026       0.050
    # teaching_ses[T.Particulièrement       -0.0131      0.026     -0.502      0.617      -0.065       0.039
    # teaching_ses[T.Particulièrement        0.0119      0.035      0.342      0.733      -0.057       0.081
    # teaching_ses[T.Public homogène,        0.0170      0.024      0.695      0.489      -0.032       0.066
    # is_researcher[T.Oui]                   0.0855      0.030      2.861      0.005       0.026       0.145
    # teaching_privpubl[T.Public]            0.1099      0.040      2.719      0.008       0.030       0.190
    # is_trainer_support[T.Oui]             -0.0074      0.022     -0.332      0.740      -0.052       0.037
    # baseline_overall_tses                  0.5184      0.092      5.642      0.000       0.336       0.701
    # yexp_teach                             0.0004      0.001      0.360      0.719      -0.002       0.002

    # Note : whether the teacher is a researcher has a significant effect on increase in self-efficacy
    # This should be interpreted with caution, as there are not many teachers who are also researchers
    #  82 are NOT researchers
    #  13 are researchers

# model comparison : higher lr_stat means that adding new variables yields an improvement
# compared to M1
for model in models[1:]:
    lr_stat, p_value, _ = results[model].compare_lr_test(results[models[0]])
    print(model)
    print('LR: ', lr_stat, 'pval: ', p_value, '\n')
# LR:  16.519365722909697 pval:  0.020772601160186505
# Adding the other covariables yields an improvement


# Repeating previous analysis with all covariables
covariables = 'baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + is_trainer_support'
models = [
    f'final_overall_tses ~ {covariables}',                                     # M1
    f'final_overall_tses ~ {covariables} + nwks',                              # M2
    f'final_overall_tses ~ {covariables} + nwks + nwks:baseline_overall_tses', # M3
    f'final_overall_tses ~ {covariables} + nwks + nwks:yexp_teach',            # M4
    f'final_overall_tses ~ {covariables} + nwks + nwks:Genre',                 # M5
]

results = {}
for model in models:
    y, X = dmatrices(model, data=df, return_type='dataframe')
    mod = sm.OLS(y, X)
    res = mod.fit()
    results[model] = res

# model comparison : higher lr_stat means that adding new variables yields an improvement
# compared to M1
for model in models[1:]:
    lr_stat, p_value, _ = results[model].compare_lr_test(results[models[0]])
    print(model)
    print('LR: ', lr_stat, 'pval: ', p_value, '\n')
    # Results :
    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + nwks
    # LR:  6.8715849620883205 pval:  0.008757694603316412
    #
    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + nwks + nwks:baseline_overall_tses
    # LR:  6.948522235646237 pval:  0.030984719424192553
    #
    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + nwks + nwks:yexp_teach
    # LR:  6.916951988859466 pval:  0.03147769766857691
    #
    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + nwks + nwks:Genre
    # LR:  9.148307136681069 pval:  0.010315026448915077
    #
    # The best model seems to be M5 again

# We fit M5 with the 3 TSES subscales
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
    #print(f"nwks : {round(res.params.loc['nwks'], 4)} [{round(c_int[0], 4)}, {round(c_int[1], 4)}], pval: {round(pval, 3)}\n")
    print(res.summary())
    # Results :

    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + nwks + nwks:Genre
    # R squared: 0.463
    # nwks : -0.0001 [-0.005, 0.0049], pval: 0.979
    #
    # final_mgmt         ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + nwks + nwks:Genre
    # R squared: 0.313
    # nwks : -0.0044 [-0.0117, 0.0029], pval: 0.234
    #
    # final_engage       ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + nwks + nwks:Genre
    # R squared: 0.388
    # nwks : -0.0029 [-0.0088, 0.0031], pval: 0.339
    #
    # final_strat        ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + nwks + nwks:Genre
    # R squared: 0.399
    # nwks : 0.0071 [0.0007, 0.0134], pval: 0.029

    # There might be a weak positive effect on instructional strategies
    # Otherwise, there seems to be no effect

    # note : for student engagement, there is a significant interaction effect with Genre :
    # nwks:Genre[T.M] 0.0075 [0.001, 0.014]  pval: 0.032
    # This means (female=0, male=1) : nkws has an effect on men in this model


# Repeating the analysis with Mixed Linear Models
# ______________ Mixed Linear Models ______________

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

# AIC comparison : the greater difference with the base model indicates the better model
for model in models:
    aic = results[model].aic
    print(model)
    print("aic:", round(aic, 2))
    # Results :

    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl
    # aic: -182.5

    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + nwks
    # aic: -181.65

    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + nwks + nwks:baseline_overall_tses
    # aic: -179.68

    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + nwks + nwks:yexp_teach
    # aic: -179.84

    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + nwks + nwks:Genre
    # aic: -183.54

    # M5 has the lower AIC and hence seems to be the better model

# We fit M5 with the 3 TSES subscales
covariables = 'baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl'
models = [
    f'final_overall_tses ~ {covariables} + nwks + nwks:Genre',
    f'final_mgmt ~ {covariables} + nwks + nwks:Genre',
    f'final_engage ~ {covariables} + nwks + nwks:Genre',
    f'final_strat ~ {covariables} + nwks + nwks:Genre',
]

for model in models:
    mod = smf.mixedlm(model, df, groups=df["user_id"], re_formula="~nwks")
    res = mod.fit(method=['lbfgs'])
    print(f"\nModèle : {model}")
    c_int = list((res.conf_int().loc['nwks']))
    pval = res.pvalues.loc['nwks']
    #print(f"nwks : {round(res.params.loc['nwks'], 4)} [{round(c_int[0], 4)}, {round(c_int[1], 4)}], pval: {round(pval, 3)}\n")
    print(res.summary())

    # Résultats
    # Modèle : final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + nwks + nwks:Genre
    #                                             Coef.  Std.Err.   z    P>|z| [0.025 0.975]
    # Intercept                                    0.236    0.109  2.170 0.030  0.023  0.449
    # Genre[T.M]                                  -0.020    0.024 -0.804 0.421 -0.068  0.028
    # teaching_ses[T.Particulièrement défavorisé  -0.002    0.024 -0.074 0.941 -0.048  0.045
    # teaching_ses[T.Particulièrement favorisé]    0.034    0.034  1.008 0.313 -0.033  0.101
    # teaching_ses[T.Public homogène, ni particul  0.023    0.023  1.001 0.317 -0.022  0.068
    # is_researcher[T.Oui]                         0.078    0.028  2.776 0.005  0.023  0.133 <-
    # teaching_privpubl[T.Privé]                   0.019    0.089  0.219 0.827 -0.155  0.194
    # teaching_privpubl[T.Public]                  0.143    0.085  1.681 0.093 -0.024  0.310
    # teaching_privpubl[T.Public&Privé]            0.060    0.129  0.466 0.641 -0.193  0.313
    # baseline_overall_tses                        0.468    0.083  5.650 0.000  0.306  0.630
    # yexp_teach                                  -0.000    0.001 -0.336 0.737 -0.002  0.002
    # nwks                                        -0.000    0.002 -0.124 0.901 -0.005  0.004 <-
    # nwks:Genre[T.M]                              0.005    0.002  1.948 0.051 -0.000  0.009 <-
    # Group Var                                    0.005
    # Group x nwks Cov                            -0.000
    # nwks Var                                     0.000
    #
    # Modèle : final_mgmt ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + nwks + nwks:Genre
    #                                             Coef.  Std.Err.   z    P>|z| [0.025 0.975]
    # Intercept                                    0.249    0.165  1.514 0.130 -0.073  0.572
    # Genre[T.M]                                  -0.026    0.037 -0.701 0.484 -0.098  0.046
    # teaching_ses[T.Particulièrement défavorisé  -0.020    0.036 -0.559 0.576 -0.090  0.050
    # teaching_ses[T.Particulièrement favorisé]   -0.002    0.050 -0.032 0.975 -0.099  0.096
    # teaching_ses[T.Public homogène, ni particul -0.000    0.034 -0.005 0.996 -0.067  0.067
    # is_researcher[T.Oui]                         0.047    0.041  1.151 0.250 -0.033  0.127
    # teaching_privpubl[T.Privé]                  -0.003    0.136 -0.020 0.984 -0.269  0.264
    # teaching_privpubl[T.Public]                  0.120    0.129  0.930 0.352 -0.133  0.374
    # teaching_privpubl[T.Public&Privé]            0.072    0.190  0.376 0.707 -0.302  0.445
    # baseline_overall_tses                        0.536    0.122  4.396 0.000  0.297  0.774
    # yexp_teach                                   0.000    0.002  0.181 0.856 -0.003  0.003
    # nwks                                        -0.004    0.003 -1.301 0.193 -0.011  0.002 <-
    # nwks:Genre[T.M]                              0.008    0.004  1.994 0.046  0.000  0.016 <-
    # Group Var                                    0.011    1.659
    # Group x nwks Cov                            -0.000    0.004
    # nwks Var                                     0.000
    # ================================================================================================================================================
    # Modèle : final_engage ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + nwks + nwks:Genre
    #                                             Coef.  Std.Err.   z    P>|z| [0.025 0.975]
    # Intercept                                    0.262    0.128  2.045 0.041  0.011  0.513
    # Genre[T.M]                                  -0.080    0.029 -2.731 0.006 -0.137 -0.022
    # teaching_ses[T.Particulièrement défavorisé ( 0.009    0.029  0.316 0.752 -0.047  0.066
    # teaching_ses[T.Particulièrement favorisé]    0.035    0.043  0.820 0.412 -0.049  0.118
    # teaching_ses[T.Public homogène, ni particuli 0.031    0.028  1.095 0.274 -0.024  0.085
    # is_researcher[T.Oui]                         0.086    0.034  2.548 0.011  0.020  0.152 <-
    # teaching_privpubl[T.Privé]                   0.012    0.104  0.113 0.910 -0.193  0.216
    # teaching_privpubl[T.Public]                  0.109    0.100  1.097 0.273 -0.086  0.305
    # teaching_privpubl[T.Public&Privé]           -0.055    0.155 -0.356 0.722 -0.358  0.248
    # baseline_overall_tses                        0.473    0.098  4.833 0.000  0.281  0.665
    # yexp_teach                                  -0.001    0.001 -0.553 0.580 -0.003  0.002
    # nwks                                        -0.004    0.003 -1.507 0.132 -0.009  0.001 <-
    # nwks:Genre[T.M]                              0.009    0.003  3.252 0.001  0.004  0.014 <-
    # Group Var                                    0.008
    # Group x nwks Cov                            -0.000
    # nwks Var                                     0.000
    #
    # Modèle : final_strat ~ baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl + nwks + nwks:Genre
    #                                             Coef.  Std.Err.   z    P>|z| [0.025 0.975]
    # Intercept                                    0.158    0.130  1.214 0.225 -0.097  0.414
    # Genre[T.M]                                   0.038    0.034  1.131 0.258 -0.028  0.104
    # teaching_ses[T.Particulièrement défavorisé ( 0.013    0.031  0.421 0.674 -0.048  0.075
    # teaching_ses[T.Particulièrement favorisé]    0.079    0.046  1.729 0.084 -0.011  0.169
    # teaching_ses[T.Public homogène, ni particuli 0.040    0.028  1.406 0.160 -0.016  0.095
    # is_researcher[T.Oui]                         0.104    0.035  2.995 0.003  0.036  0.173 <-
    # teaching_privpubl[T.Privé]                   0.060    0.110  0.544 0.586 -0.155  0.275
    # teaching_privpubl[T.Public]                  0.207    0.097  2.135 0.033  0.017  0.397
    # teaching_privpubl[T.Public&Privé]            0.173    0.161  1.071 0.284 -0.143  0.489
    # baseline_overall_tses                        0.430    0.123  3.508 0.000  0.190  0.670
    # yexp_teach                                  -0.000    0.001 -0.150 0.881 -0.003  0.002
    # nwks                                         0.007    0.003  2.205 0.027  0.001  0.012 <-
    # nwks:Genre[T.M]                             -0.002    0.004 -0.481 0.630 -0.010  0.006 <-
    # Group Var                                    0.011
    # Group x nwks Cov                            -0.001
    # nwks Var                                     0.000
    # ================================================================================================================================================

    # We find again a slight effect on the instructional strategies subscale
    # and an interaction for student engagement (nwks has an effect on male participants)


# Investigating a potential interaction between nwks and is_researcher
covariables = 'baseline_overall_tses + yexp_teach + Genre + teaching_ses + is_researcher + teaching_privpubl'
models = [
    f'final_overall_tses ~ {covariables} + nwks + nwks:is_researcher',
    f'final_mgmt ~ {covariables} + nwks + nwks:is_researcher',
    f'final_engage ~ {covariables} + nwks + nwks:is_researcher',
    f'final_strat ~ {covariables} + nwks + nwks:is_researcher',
]

for model in models:
    mod = smf.mixedlm(model, df, groups=df["user_id"], re_formula="~nwks")
    res = mod.fit(method=['lbfgs'])
    print(f"\nModèle : {model}")
    c_int = list((res.conf_int().loc['nwks']))
    pval = res.pvalues.loc['nwks']
    #print(f"nwks : {round(res.params.loc['nwks'], 4)} [{round(c_int[0], 4)}, {round(c_int[1], 4)}], pval: {round(pval, 3)}\n")
    print(res.summary())
    # Résultat : pas d'effet d'interaction entre nwks et is_researcher

