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
#
#
## model comparison : higher lr_stat means that adding new variables yields an improvement
## compared to M1
#for model in models[1:]:
#    lr_stat, p_value, _ = results[model].compare_lr_test(results[models[0]])
#    print(model)
#    print('LR: ', lr_stat, 'pval: ', p_value, '\n')
#    # Results :
#
#    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks
#    # LR:  4.810665142679568 pval:  0.02828412616677737
#
#    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:baseline_overall_tses
#    # LR:  5.0312016471714855 pval:  0.08081434245422556
#
#    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:yexp_teach
#    # LR:  4.974783376204982 pval:  0.08312650390309326
#
#    # final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:Genre
#    # LR:  5.808773163502508 pval:  0.054782384811339284
#
#    # The best model seems to be M5
#
## We fit M5 with the 3 TSES subscales
#models = [
#    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:Genre',
#    'final_mgmt ~ baseline_mgmt + yexp_teach + Genre + nwks + nwks:Genre',
#    'final_engage ~ baseline_engage + yexp_teach + Genre + nwks + nwks:Genre',
#    'final_strat ~ baseline_strat + yexp_teach + Genre + nwks + nwks:Genre',
#]
#
#for model in models:
#    y, X = dmatrices(model, data=df, return_type='dataframe')
#    mod = sm.OLS(y, X)
#    res = mod.fit()
#    print(model)
#    print(f"R squared: {round(res.rsquared, 3)}")
#    c_int = list((res.conf_int().loc['nwks']))
#    pval = res.pvalues.loc['nwks']
#    print(f"nwks : {round(res.params.loc['nwks'], 4)} [{round(c_int[0], 4)}, {round(c_int[1], 4)}], pval: {round(pval, 3)}\n")
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

#models = [
#    # Regression models from pagnotta et al.
#    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre',                                      # M1
#
#    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks',                               # M2
#
#    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:baseline_overall_tses',  # M3
#    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:yexp_teach',             # M4
#    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:Genre',                  # M5
#]
#
#results = {}
#for model in models:
#    mod = smf.mixedlm(model, df, groups=df["user_id"], re_formula="~nwks")
#    res = mod.fit(method=['lbfgs'], reml=False)
#    results[model] = res
#
#
#for model in models:
#    aic = results[model].aic
#    print(model)
#    print("aic:", round(aic, 2))
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

    # Maximum AIC difference is between M1 and M3. So M3 is the preferred model


# We fit M3 with the 3 TSES subscales

#models = [
#    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:baseline_overall_tses',
#    'final_mgmt ~ baseline_mgmt + yexp_teach + Genre + nwks + nwks:baseline_mgmt',
#    'final_engage ~ baseline_engage + yexp_teach + Genre + nwks + nwks:baseline_engage',
#    'final_strat ~ baseline_strat + yexp_teach + Genre + nwks + nwks:baseline_strat',
#]
#
#for model in models:
#    mod = smf.mixedlm(model, df, groups=df["user_id"], re_formula="~nwks")
#    res = mod.fit(method=['lbfgs'])
#    print(f"\nModèle : {model}")
#    c_int = list((res.conf_int().loc['nwks']))
#    pval = res.pvalues.loc['nwks']
#    #print(f"nwks : {round(res.params.loc['nwks'], 4)} [{round(c_int[0], 4)}, {round(c_int[1], 4)}], pval: {round(pval, 3)}\n")
#    print(res.summary())
#
#    # Résultats
#    # Modèle : final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:baseline_overall_tses
#    #                            Coef.  Std.Err.   z    P>|z| [0.025 0.975]
#    # ---------------------------------------------------------------------
#    # Intercept                   0.395    0.086  4.570 0.000  0.226  0.565
#    # Genre[T.M]                  0.004    0.019  0.240 0.810 -0.032  0.041
#    # baseline_overall_tses       0.460    0.115  3.998 0.000  0.235  0.686
#    # yexp_teach                 -0.001    0.001 -0.556 0.578 -0.003  0.001
#    # nwks                       -0.002    0.011 -0.146 0.884 -0.022  0.019 <-
#    # nwks:baseline_overall_tses  0.004    0.013  0.279 0.781 -0.022  0.030 <-
#    # Group Var                   0.006
#    # Group x nwks Cov           -0.000
#    # nwks Var                    0.000
#    #
#    # Modèle : final_mgmt ~ baseline_mgmt + yexp_teach + Genre + nwks + nwks:baseline_mgmt
#    #                    Coef.  Std.Err.   z    P>|z| [0.025 0.975]
#    # -------------------------------------------------------------
#    # Intercept           0.255    0.091  2.817 0.005  0.078  0.433
#    # Genre[T.M]         -0.001    0.024 -0.044 0.965 -0.048  0.046
#    # baseline_mgmt       0.667    0.124  5.382 0.000  0.424  0.909
#    # yexp_teach         -0.001    0.001 -0.594 0.553 -0.003  0.002
#    # nwks                0.010    0.008  1.191 0.234 -0.006  0.026 <-
#    # nwks:baseline_mgmt -0.010    0.011 -0.915 0.360 -0.032  0.011 <-
#    # Group Var           0.009
#    # Group x nwks Cov   -0.000
#    # nwks Var            0.000
#    #
#    # Modèle : final_engage ~ baseline_engage + yexp_teach + Genre + nwks + nwks:baseline_engage
#    #                      Coef.  Std.Err.   z    P>|z| [0.025 0.975]
#    # ---------------------------------------------------------------
#    # Intercept             0.462    0.077  5.996 0.000  0.311  0.613
#    # Genre[T.M]           -0.020    0.022 -0.944 0.345 -0.063  0.022
#    # baseline_engage       0.360    0.103  3.498 0.000  0.158  0.561
#    # yexp_teach           -0.000    0.001 -0.387 0.699 -0.003  0.002
#    # nwks                 -0.006    0.012 -0.509 0.610 -0.030  0.018 <-
#    # nwks:baseline_engage  0.007    0.015  0.467 0.641 -0.023  0.037 <-
#    # Group Var             0.010
#    # Group x nwks Cov     -0.001
#    # nwks Var              0.000
#    #
#    # Modèle : final_strat ~ baseline_strat + yexp_teach + Genre + nwks + nwks:baseline_strat
#    #                     Coef.  Std.Err.   z    P>|z| [0.025 0.975]
#    # --------------------------------------------------------------
#    # Intercept            0.491    0.106  4.621 0.000  0.283  0.700
#    # Genre[T.M]           0.028    0.024  1.173 0.241 -0.019  0.075
#    # baseline_strat       0.337    0.131  2.569 0.010  0.080  0.594
#    # yexp_teach          -0.001    0.001 -0.834 0.404 -0.003  0.001
#    # nwks                -0.006    0.013 -0.455 0.649 -0.032  0.020 <-
#    # nwks:baseline_strat  0.011    0.015  0.730 0.465 -0.018  0.040 <-
#    # Group Var            0.011    0.397
#    # Group x nwks Cov    -0.000    0.005
#    # nwks Var             0.000    0.000
#
#    # When adjusting for random effects, there seems to be no effect of nwks on TSE whatsoever.
#
#

# Comparing linear models and mixed models with anova
model = 'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:baseline_overall_tses'

#y, X = dmatrices(model, data=df, return_type='dataframe')
mod = smf.ols(model, data=df)
res = mod.fit()

modmixed = smf.mixedlm(model, df, groups=df["user_id"], re_formula="~nwks")
resmixed = modmixed.fit(method=['lbfgs'])

anova = sm.stats.anova_lm(res, res)
print(anova)
#TODO: Look at other confounding variables !

