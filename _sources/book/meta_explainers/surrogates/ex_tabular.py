# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Surrogate Explainer of Tabular Data #
#
# This exercise allows you to experiment with the parameterisation of a linear
# surrogate for tabular data.
#
# To launch this exercise as a Jupyter Notebook with MyBinder or Google Colab
# use the *Launch Menu* that appears after hovering the mouse cursor over the
# {fa}`rocket` icon shown in the top bar.

# %% tags=['remove-cell']
import fatf
import fatf.utils.data.augmentation as fatf_augmentation

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn import datasets

import ipywidgets as widgets
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn')
# %matplotlib inline

# %% tags=['remove-cell']
def plot_pred(tuple_list, ax=None):
    x = [i[0] for i in tuple_list[::-1]]
    y = [i[1] for i in tuple_list[::-1]]

    if ax is None:
        ax = plt
        ax.figure(figsize=(4,4))
        ax.xlim([0, 1.20])
        ax.ylim([-.5, len(x) - .5])
    else:
        ax.set_xlim([0, 1.30])
        ax.set_ylim([-.5, len(x) - .5])
        ax.grid(False, axis='y')
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    ax.barh(x, y, height=.5)
    for i, v in enumerate(y):
        ax.text(v + .02, i + .0, '{:.4f}'.format(v), fontweight='bold', fontsize=18)

# %% [markdown] tags=['remove-cell']
# ## Surrogate Explainers of Tabular Data -- Classifying Iris Flowers ##
#
# <img width="30%" align="middle" src="iris.png" alt="Iris Data Set" style="display: block; margin-left: auto; margin-right: auto;">
# <img width="70%" align="middle" src="iris-classes.jpeg" alt="Iris Classes" style="display: block; margin-left: auto; margin-right: auto;">

# %% tags=['remove-cell']
iris = datasets.load_iris()
X = iris.data  #[:, :2]  # we only take the first two features.
Y = iris.target

PFI_REPEATS = 30

# %% tags=['remove-cell']
logreg = LogisticRegression(C=1e5)
logreg.fit(X, Y)

# %% tags=['remove-cell']
explained_class = {j:i for i, j in enumerate(iris.target_names)}
explained_instances = {'setosa': np.array([5, 3.5, 1.5, 0.25]).astype(X.dtype),
                       'versicolor': np.array([5.5, 2.75, 4.5, 1.25]).astype(X.dtype),
                       'virginica': np.array([7, 3, 5.5, 2.25]).astype(X.dtype)}
petal_length_bins = [1, 2, 3, 4, 5, 6, 7]
petal_width_bins = [0, .5, 1, 1.5, 2, 2.5]

IRIS_SAMPLED = False

# %% tags=['remove-cell']
lime_data = {}
iris_augmenter = fatf_augmentation.Mixup(X, ground_truth=Y)
for class_name, data_point in explained_instances.items():
    if IRIS_SAMPLED:
        iris_sampled_data = iris_augmenter.sample(data_point, samples_number=50000)
    else:
        iris_sampled_data = X.copy()

    iris_sampled_data_probabilities = logreg.predict_proba(iris_sampled_data)
    lime_data[class_name] = (iris_sampled_data, iris_sampled_data_probabilities)

# %% tags=['remove-cell']
def explain_lime(pl_bounds, pw_bounds, explained_point, explained_point_class, explained_class):
    data_ = lime_data[explained_point_class][0]
    preds_ = lime_data[explained_point_class][1][:, explained_class]

    # digitize data
    data_dig = np.vstack([
        np.digitize(data_[:, 2], pl_bounds),
        np.digitize(data_[:, 3], pw_bounds)
    ]).T
    # digitize point
    point_dig = np.array([
        np.digitize(explained_point[2], pl_bounds),
        np.digitize(explained_point[3], pw_bounds)
        ])
    #
    binary_data = (data_dig == point_dig).astype(np.int8)
    # print(np.unique(binary_data, axis=0))

    # train ridge
    ri = Ridge()
    ri.fit(binary_data, preds_)
    # return coefficients
    return ri.coef_

# %% tags=['remove-cell']
def plot_lime(instance_class, explained_class, instance, petal_length_range, petal_width_range, explanation):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 6))
    fig.patch.set_alpha(0)
    fig.suptitle('Explained instance: {}    |    Explained class: {}'.format(instance_class, explained_class), fontsize=18)

    # plot /petal length (cm)/ vs /petal width/
    x_name, y_name = 'petal length (cm)', 'petal width (cm)'
    x_ind, y_ind = iris.feature_names.index(x_name), iris.feature_names.index(y_name)
    x_min, x_max = X[:, x_ind].min() - .5, X[:, x_ind].max() + .5
    y_min, y_max = X[:, y_ind].min() - .5, X[:, y_ind].max() + .5
    #
    ax_l.scatter(X[:, x_ind], X[:, y_ind], c=Y, cmap=plt.cm.Set1, edgecolor='k')
    ax_l.set_xlabel(x_name, fontsize=18)
    ax_l.set_ylabel(y_name, fontsize=18)
    #
    ax_l.set_xlim(x_min, x_max)
    ax_l.set_ylim(y_min, y_max)
    # plt.xticks(())
    # plt.yticks(())
    ax_l.scatter(instance[x_ind], instance[y_ind], c='yellow', marker='*', s=500, edgecolor='k')
    ax_l.vlines(petal_length_range, -1, 10, linewidth=3)
    ax_l.hlines(petal_width_range, -1, 10, linewidth=3)
    #
    ax_l.tick_params(axis='x', labelsize=18)
    ax_l.tick_params(axis='y', labelsize=18)


    x_dig_ = np.digitize(instance[x_ind], petal_length_range)
    x_dig_list_ = ['-inf'] + [str(i) for i in petal_length_range] + ['+inf']
    y_dig_ = np.digitize(instance[y_ind], petal_width_range)
    y_dig_list_ = ['-inf'] + [str(i) for i in petal_width_range] + ['+inf']
    x = ['{}\n{} < ... <= {}'.format(iris.feature_names[2], x_dig_list_[x_dig_], x_dig_list_[x_dig_+1]),
         '{}\n{} < ... <= {}'.format(iris.feature_names[3], y_dig_list_[y_dig_], y_dig_list_[y_dig_+1])]
    #
    y = [abs(i) for i in explanation]
    c = ['green' if i>=0 else 'red' for i in explanation]

    ax_r.set_xlim([0, 1.20])
    ax_r.set_ylim([-.5, len(x) - .5])
    ax_r.grid(False, axis='y')
    ax_r.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    ax_r.barh(x, y, height=.5, color=c)
    ax_r.set_yticklabels([])
    for i, v in enumerate(y):
        ax_r.text(v + .02, i + .15, '{:.4f}'.format(v), fontweight='bold', fontsize=18)

        ax_r.text(v + .02, i - .2, x[i], fontweight='bold', fontsize=18)

    # highlight explained spot
    x_dig_list_val_ = [0] + [i for i in petal_length_range] + [8]
    ax_l.axvspan(x_dig_list_val_[x_dig_], x_dig_list_val_[x_dig_+1],
            facecolor='blue', alpha=0.2)
    y_dig_list_val_ = [-.5] + [i for i in petal_width_range] + [3.5]
    ax_l.axhspan(y_dig_list_val_[y_dig_], y_dig_list_val_[y_dig_+1],
            facecolor='yellow', alpha=0.2)
    
    ax_r.tick_params(axis='x', labelsize=18)
#     ax_r.tick_params(axis='y', labelsize=18)

    plt.show()

# %% tags=['remove-cell']
# select 1 fo three points
# select to explain a class
# select two thresholds for the segmentation
lime_instance_toggle = widgets.ToggleButtons(
    options=list(explained_instances.keys()),
    description='Instance:',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    # tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
#     icons=['check'] * 3
)
lime_class_toggle = widgets.ToggleButtons(
    options=list(explained_class.keys()),
    description='Class:',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    # tooltips=['Description of slow', 'Description of regular', 'Description of fast'],
#     icons=['check'] * 3
)
iris_petal_length_slider = widgets.FloatRangeSlider(
    value=[2, 3],
    min=1,
    max=7,
    step=1,
    description='Length (x):',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.0f',
)
iris_petal_width_slider = widgets.FloatRangeSlider(
    value=[.5, 1],
    min=0,
    max=2.5,
    step=0.5,
    description='Width (y):',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)
lime_explain_button = widgets.Button(
    description='Explain!',
    disabled=False,
    button_style='info', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Explain',
    icon='check'
)
lime_explain_out = widgets.Output()

# %% tags=['remove-cell']
def explain_lime_button_func(obj):
    instance_class_ = lime_instance_toggle.value
    instance_ = explained_instances[instance_class_]

    explained_class_ = lime_class_toggle.value
    explained_class_id_ = explained_class[explained_class_]

    petal_length_range_ = iris_petal_length_slider.value
    petal_width_range_ = iris_petal_width_slider.value

    explanation_ = explain_lime(petal_length_range_, petal_width_range_,
                                instance_, instance_class_,
                                explained_class_id_)

    with lime_explain_out:
        lime_explain_out.clear_output(wait=True)
        plot_lime(instance_class_, explained_class_, instance_, petal_length_range_, petal_width_range_, explanation_)
        plt.show()

    # if problog_programme:
        # pass
    # else:
        # d = {'error': 'No explanation available'}
    # return d
lime_explain_button.on_click(explain_lime_button_func)
lime_explain_button._click_handlers(lime_explain_button)  # pre-click the button

# %% tags=['remove-cell']
surrogate_tabular_explainer = widgets.VBox([lime_instance_toggle, lime_class_toggle, iris_petal_length_slider, iris_petal_width_slider, lime_explain_button, lime_explain_out])

# %% [markdown] tags=['remove-cell']
# ## Explainer Demo ##

# %% tags=['remove-cell']
surrogate_tabular_explainer
