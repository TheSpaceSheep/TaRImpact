import numpy as np
import statsmodels
import statsmodels.api as sm
from patsy import dmatrices
from data_loading import load_data
from utils import *

import matplotlib.pyplot as plt
import seaborn as sns


df = get_processed_tses()

print(df.columns)
print(len(df))
sns.histplot(data=df, x='nwks', palette='cool', binwidth=1)
plt.show()

# Toy example : does yexp predict baseline TSE ?
#y, X = dmatrices('baseline_mean ~ yexp_teach', data=df, return_type='dataframe')
#
#mod = sm.OLS(y, X)
#res= mod.fit()


# TODO: separate by community
# TODO: separate by TSES subscales
# TODO: separate by high/low baseline

models = [
    #'final_overall_tses ~ baseline_overall_tses',
    # Regression models from pagnotta et al.
    #'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre',
    #'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks',
    #'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:baseline_overall_tses',
    #'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:yexp_teach',
    'final_overall_tses ~ baseline_overall_tses + yexp_teach + Genre + nwks + nwks:Genre',

    #'final_mgmt ~ baseline_mgmt + yexp_teach + Genre',
    #'final_mgmt ~ baseline_mgmt + yexp_teach + Genre + nwks',
    #'final_mgmt ~ baseline_mgmt + yexp_teach + Genre + nwks + nwks:baseline_mgmt',
    #'final_mgmt ~ baseline_mgmt + yexp_teach + Genre + nwks + nwks:yexp_teach',
    'final_mgmt ~ baseline_mgmt + yexp_teach + Genre + nwks + nwks:Genre',

    #'final_engage ~ baseline_engage + yexp_teach + Genre',
    #'final_engage ~ baseline_engage + yexp_teach + Genre + nwks',
    #'final_engage ~ baseline_engage + yexp_teach + Genre + nwks + nwks:baseline_engage',
    #'final_engage ~ baseline_engage + yexp_teach + Genre + nwks + nwks:yexp_teach',
    'final_engage ~ baseline_engage + yexp_teach + Genre + nwks + nwks:Genre',

    #'final_strat ~ baseline_strat + yexp_teach + Genre',
    #'final_strat ~ baseline_strat + yexp_teach + Genre + nwks',
    #'final_strat ~ baseline_strat + yexp_teach + Genre + nwks + nwks:baseline_strat',
    #'final_strat ~ baseline_strat + yexp_teach + Genre + nwks + nwks:yexp_teach',
    'final_strat ~ baseline_strat + yexp_teach + Genre + nwks + nwks:Genre',
]


results = {}  # for later model comparison
for model in models:
    y, X = dmatrices(model, data=df, return_type='dataframe')
    mod = sm.OLS(y, X)
    res= mod.fit()
    results[model] = res
    print(f"\n\nModèle : {model}", end="")
    print(res.summary())

#for model in models[1:]:
#    print(results[model].compare_lr_test(
#        results[models[0]]
#    ))
#for model in models[2:]:
#    print(results[model].compare_lr_test(
#        results[models[1]]
#    ))
