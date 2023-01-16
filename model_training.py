# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 21:53:29 2018

@author: J Wolf
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from basic_linear_model import BasicLinearModel


def time_to_string(t):
    t = int(t)
    h = t // 3600
    m = (t % 3600) // 60
    s = (t % 3600) % 60

    h = "0" + str(h) if h < 10 else str(h)
    m = "0" + str(m) if m < 10 else str(m)
    s = "0" + str(s) if s < 10 else str(s)

    return h + ":" + m + ":" + s


def string_to_time(time_string):
    value_strings = time_string.split(":")
    value = 0
    if len(value_strings) == 2:
        value += int(value_strings[0]) * 60 + int(value_strings[1])
    else:
        if len(value_strings[2]) > 2:
            if value_strings[2][-2] == "P":
                value += 12 * 3600
            value_strings[2] = value_strings[2][:2]
        value += (
            (int(value_strings[0]) % 12) * 3600
            + int(value_strings[1]) * 60
            + int(value_strings[2])
        )
    return value


col_list = [
    "sex",
    "age group",
    "starting time",
    "5k split time",
    "10k split time",
    "15k split time",
    "20k split time",
    "Half split time",
    "25k split time",
    "30k split time",
    "35k split time",
    "40k split time",
    "Finish split time",
]

data = pd.read_csv(
    "./data_files/clean_chicago_data.csv", header=0, usecols=col_list, low_memory=False
)
bq_data = pd.read_csv(
    ".data_files/clean_chicago_bq_data.csv",
    header=0,
    usecols=col_list,
    low_memory=False,
)

X_train, X_test = train_test_split(data, test_size=0.2)
male_train, male_test = train_test_split(data[data["sex"] == 0], test_size=0.2)
female_train, female_test = train_test_split(data[data["sex"] == 1], test_size=0.2)
masters_train, masters_test = train_test_split(
    data[data["age group"] >= 6], test_size=0.2
)
bq_train, bq_test = train_test_split(bq_data, test_size=0.2)


"""
##########  FULL DATA NEURAL NETWORK TRAINING  ##########
training_start_time = time.time()
networks = [MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100, 100)) for _ in range(10)]
for i in range(10):
    train_columns = X_train.columns[:i+3]
    target_columns = X_train.columns[i+3:]
    networks[i].fit(X_train[train_columns], X_train[target_columns])
    print('nn{0} score:'.format(i), networks[i].score(X_test[train_columns], X_test[target_columns]))


training_finish_time = time.time()
print('TOTAL TRAINING TIME OF NNS:', training_finish_time - training_start_time)
###########  END OF FULL DATA NEURAL NETWORK TRAINING  ###########



##########  BQ DATA NEURAL NETWORK TRAINING  ##########
training_start_time = time.time()
bq_networks = [MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100, 100, 100, 100)) for _ in range(10)]
for i in range(10):
    train_columns = bq_train.columns[:i+3]
    target_columns = bq_train.columns[i+3:]
    bq_networks[i].fit(bq_train[train_columns], bq_train[target_columns])
    print('bq_nn{0} score:'.format(i), bq_networks[i].score(bq_test[train_columns], bq_test[target_columns]))

training_finish_time = time.time()
print('TOTAL TRAINING TIME OF BQ NNS:', training_finish_time - training_start_time)
###########  END OF BQ DATA NEURAL NETWORK TRAINING  ###########



##########  FULL DATA GRADIENT BOOSTING TRAINING  ##########
training_start_time = time.time()
g_boosts = [GradientBoostingRegressor() for _ in range(10)]
for i in range(10):
    train_columns = X_train.columns[:i+3]
    g_boosts[i].fit(X_train[train_columns], X_train['Finish split time'])
    print('g_boost{0} score:'.format(i), g_boosts[i].score(X_test[train_columns], X_test['Finish split time']))

training_finish_time = time.time()
print('TOTAL TRAINING TIME OF G BOOSTS:', training_finish_time - training_start_time)
##########  END OF FULL DATA GRADIENT BOOSTING TRAINING  ##########

    
    
##########  BQ DATA GRADIENT BOOSTING TRAINING  ##########
training_start_time = time.time()
bq_g_boosts = [GradientBoostingRegressor() for _ in range(10)]
for i in range(10):
    train_columns = bq_train.columns[:i+3]
    bq_g_boosts[i].fit(bq_train[train_columns], bq_train['Finish split time'])
    print('bq_g_boost{0} score:'.format(i), bq_g_boosts[i].score(bq_test[train_columns], bq_test['Finish split time']))

training_finish_time = time.time()
print('TOTAL TRAINING TIME OF BQ G BOOSTS:', training_finish_time - training_start_time)
##########  END OF BQ DATA GRADIENT BOOSTING TRAINING  ##########

#save the models
for i in range(10):
    joblib.dump(networks[i], './models/nn{0}.pkl'.format(i))
    joblib.dump(bq_networks[i], './models/bq_nn{0}.pkl'.format(i))
    joblib.dump(g_boosts[i], './models/g_boost{0}.pkl'.format(i))
    joblib.dump(bq_g_boosts[i], './models/bq_g_boost{0}.pkl'.format(i))
"""

#########  Load the models  ##########
networks = []
bq_networks = []
g_boosts = []
bq_g_boosts = []

for i in range(10):
    networks.append(joblib.load("./models/nn{0}.pkl".format(i)))
    bq_networks.append(joblib.load("./models/bq_nn{0}.pkl".format(i)))
    g_boosts.append(joblib.load("./models/g_boost{0}.pkl".format(i)))
    bq_g_boosts.append(joblib.load("./models/bq_g_boost{0}.pkl".format(i)))
##########  End of loading the models ###########


def predict_runner(runner):
    # check formatting of times for runner
    if len(runner) > 3:
        for i in range(3, len(runner)):
            if isinstance(runner[i], str):
                runner[i] = string_to_time(runner[i])

    # print the predictions of each network
    for i in range(min(len(runner) - 2, 10)):
        print("NETWORK {0} PREDICTIONS:".format(i))
        x = (
            [time_to_string(y) for y in networks[i].predict([runner[: 3 + i]])[0]]
            if i < 9
            else [time_to_string(y) for y in networks[i].predict([runner[: 3 + i]])]
        )
        print(*x)


def bq_predict_runner(runner):
    # check formatting of times for runner
    if len(runner) > 3:
        for i in range(3, len(runner)):
            if isinstance(runner[i], str):
                runner[i] = string_to_time(runner[i])

    # print the predictions of each network
    for i in range(min(len(runner) - 2, 10)):
        print("BQ NETWORK {0} PREDICTIONS:".format(i))
        x = (
            [time_to_string(y) for y in bq_networks[i].predict([runner[: 3 + i]])[0]]
            if i < 9
            else [time_to_string(y) for y in bq_networks[i].predict([runner[: 3 + i]])]
        )
        print(*x)


def bq_g_boost_predict_runner(runner):
    # check formatting of times for runner
    if len(runner) > 3:
        for i in range(3, len(runner)):
            if isinstance(runner[i], str):
                runner[i] = string_to_time(runner[i])

    # print the predictions of each network
    for i in range(min(len(runner) - 2, 10)):
        print("BQ G BOOST {0} PREDICTIONS:".format(i))
        x = (
            [time_to_string(y) for y in bq_g_boosts[i].predict([runner[: 3 + i]])]
            if i < 9
            else [time_to_string(y) for y in bq_g_boosts[i].predict([runner[: 3 + i]])]
        )
        print(*x)


def basic_predict_runner(runner):
    # check to ignore sex, age group, and starting time columns
    if runner[0] in [0, 1]:
        runner = runner[3:]
    for i in range(len(runner)):
        if isinstance(runner[i], str):
            runner[i] = string_to_time(runner[i])

    for i in range(1, len(runner)):
        print("LINEAR MODEL", col_list[i + 2], "PREDICTIONS:")

        x = [
            time_to_string(y)
            for y in clf.predict(np.array(runner[:i]).reshape(1, -1))[0]
        ]
        print(*x)


clf = BasicLinearModel()


########## TESTING THE MODELS FINISH TIMES AGAINST GENERAL RUNNERS
lin_r2 = [0]
nn_r2 = [
    r2_score(
        X_test["Finish split time"], networks[0].predict(X_test[col_list[:3]])[:, -1]
    )
]
bq_nn_r2 = [
    r2_score(
        X_test["Finish split time"], bq_networks[0].predict(X_test[col_list[:3]])[:, -1]
    )
]
g_boost_r2 = [
    r2_score(
        X_test["Finish split time"],
        g_boosts[0].predict(X_test[col_list[:3]]).reshape(-1, 1),
    )
]
bq_g_boost_r2 = [
    r2_score(
        X_test["Finish split time"],
        bq_g_boosts[0].predict(X_test[col_list[:3]]).reshape(-1, 1),
    )
]

for i in range(4, len(col_list)):
    # print(col_list[i])
    y_true = X_test.iloc[:, i:].values
    lin_pred = clf.predict(X_test.iloc[:, 3:i].values)
    # print('Linear model', r2_score(y_true, lin_pred))
    # print('Linear model FINISH TIME', r2_score(y_true[:,-1], lin_pred[:,-1]))
    lin_r2.append(r2_score(y_true[:, -1], lin_pred[:, -1]))

    train_cols = col_list[:i]
    pred_cols = col_list[i:]
    nn_pred = networks[i - 3].predict(X_test[train_cols])
    bq_nn_pred = bq_networks[i - 3].predict(X_test[train_cols])
    g_boost_pred = g_boosts[i - 3].predict(X_test[train_cols]).reshape(-1, 1)
    bq_g_boost_pred = bq_g_boosts[i - 3].predict(X_test[train_cols]).reshape(-1, 1)

    if i == len(col_list) - 1:
        nn_pred = nn_pred.reshape(-1, 1)
        bq_nn_pred = bq_nn_pred.reshape(-1, 1)

    # print(col_list[i], 'neural model', networks[i-3].score(X_test[train_cols], X_test[pred_cols]))
    # print('neural model FINISH TIME', r2_score(y_true[:,-1], nn_pred[:,-1]))
    nn_r2.append(r2_score(y_true[:, -1], nn_pred[:, -1]))
    # print('BQ neural model FINISH TIME', r2_score(y_true[:,-1], bq_nn_pred[:,-1]))
    bq_nn_r2.append(r2_score(y_true[:, -1], bq_nn_pred[:, -1]))
    # print('g boost model FINISH TIME', r2_score(y_true[:,-1], g_boost_pred[:,-1]))
    g_boost_r2.append(r2_score(y_true[:, -1], g_boost_pred[:, -1]))
    # print('BQ g boost model FINISH TIME', r2_score(y_true[:,-1], bq_g_boost_pred[:,-1]))
    bq_g_boost_r2.append(r2_score(y_true[:, -1], bq_g_boost_pred[:, -1]))

x_ticks = ["start", "5k", "10k", "15k", "20k", "Half", "25k", "30k", "35k", "40k"]
plt.title("R2_scores for finish time - All runners")
plt.plot(range(10), lin_r2)
plt.plot(range(10), nn_r2)
plt.plot(range(10), bq_nn_r2)
plt.plot(range(10), g_boost_r2)
plt.plot(range(10), bq_g_boost_r2)
plt.xticks(range(10), x_ticks)
plt.legend(["lin", "nn", "bq_nn", "g_boost", "bq_g_boost"])
plt.savefig("./results/general_finish_all_models.png")
plt.show()


plt.title("R2_scores for finish time - All runners")
plt.plot(range(10), lin_r2)
plt.plot(range(10), nn_r2)
plt.plot(range(10), g_boost_r2)
plt.xticks(range(10), x_ticks)
plt.legend(["lin", "nn", "g_boost"])
plt.savefig("./results/general_finish_no_bq_models.png")
plt.show()

print("GENERAL RESULTS:")
print("Linear model:", "&".join([format(x, ".3f") for x in lin_r2]))
print("Neural Network:", "&".join([format(x, ".3f") for x in nn_r2]))
print("Gradient boosting regressor:", "&".join([format(x, ".3f") for x in g_boost_r2]))
print("BQ Neural Network:", "&".join([format(x, ".3f") for x in bq_nn_r2]))
print(
    "BQ Gradient boosting regressor:",
    "&".join([format(x, ".3f") for x in bq_g_boost_r2]),
)

######## TESTING AGAINST BQ RUNNERS #########
lin_r2 = [0]
nn_r2 = [
    r2_score(
        bq_test["Finish split time"], networks[0].predict(bq_test[col_list[:3]])[:, -1]
    )
]
bq_nn_r2 = [
    r2_score(
        bq_test["Finish split time"],
        bq_networks[0].predict(bq_test[col_list[:3]])[:, -1],
    )
]
g_boost_r2 = [
    r2_score(
        bq_test["Finish split time"],
        g_boosts[0].predict(bq_test[col_list[:3]]).reshape(-1, 1),
    )
]
bq_g_boost_r2 = [
    r2_score(
        bq_test["Finish split time"],
        bq_g_boosts[0].predict(bq_test[col_list[:3]]).reshape(-1, 1),
    )
]

for i in range(4, len(col_list)):
    # print(col_list[i])
    y_true = bq_test.iloc[:, i:].values
    lin_pred = clf.predict(bq_test.iloc[:, 3:i].values)
    # print('Linear model', r2_score(y_true, lin_pred))
    # print('Linear model FINISH TIME', r2_score(y_true[:,-1], lin_pred[:,-1]))
    lin_r2.append(r2_score(y_true[:, -1], lin_pred[:, -1]))

    train_cols = col_list[:i]
    pred_cols = col_list[i:]
    nn_pred = networks[i - 3].predict(bq_test[train_cols])
    bq_nn_pred = bq_networks[i - 3].predict(bq_test[train_cols])
    g_boost_pred = g_boosts[i - 3].predict(bq_test[train_cols]).reshape(-1, 1)
    bq_g_boost_pred = bq_g_boosts[i - 3].predict(bq_test[train_cols]).reshape(-1, 1)

    if i == len(col_list) - 1:
        nn_pred = nn_pred.reshape(-1, 1)
        bq_nn_pred = bq_nn_pred.reshape(-1, 1)

    # print(col_list[i], 'neural model', networks[i-3].score(X_test[train_cols], X_test[pred_cols]))
    # print('neural model FINISH TIME', r2_score(y_true[:,-1], nn_pred[:,-1]))
    nn_r2.append(r2_score(y_true[:, -1], nn_pred[:, -1]))
    # print('BQ neural model FINISH TIME', r2_score(y_true[:,-1], bq_nn_pred[:,-1]))
    bq_nn_r2.append(r2_score(y_true[:, -1], bq_nn_pred[:, -1]))
    # print('g boost model FINISH TIME', r2_score(y_true[:,-1], g_boost_pred[:,-1]))
    g_boost_r2.append(r2_score(y_true[:, -1], g_boost_pred[:, -1]))
    # print('BQ g boost model FINISH TIME', r2_score(y_true[:,-1], bq_g_boost_pred[:,-1]))
    bq_g_boost_r2.append(r2_score(y_true[:, -1], bq_g_boost_pred[:, -1]))


plt.title("R2_scores for finish time - BQ runners")
plt.plot(range(10), lin_r2)
plt.plot(range(10), nn_r2)
plt.plot(range(10), bq_nn_r2)
plt.plot(range(10), g_boost_r2)
plt.plot(range(10), bq_g_boost_r2)
plt.xticks(range(10), x_ticks)
plt.legend(["lin", "nn", "bq_nn", "g_boost", "bq_g_boost"])
plt.savefig("./results/bq_finish_all_models.png")
plt.show()


plt.title("R2_scores for finish time - BQ runners")
plt.plot(range(10), lin_r2)
plt.plot(range(10), bq_nn_r2)
plt.plot(range(10), bq_g_boost_r2)
plt.xticks(range(10), x_ticks)
plt.legend(["lin", "bq_nn", "bq_g_boost"])
plt.savefig("./results/bq_finish_bq_models_only.png")
plt.show()


print("BQ RESULTS:")
print("Linear model:", "&".join([format(x, ".3f") for x in lin_r2]))
print("BQ Neural Network:", "&".join([format(x, ".3f") for x in bq_nn_r2]))
print(
    "BQ Gradient boosting regressor:",
    "&".join([format(x, ".3f") for x in bq_g_boost_r2]),
)
print("Neural Network:", "&".join([format(x, ".3f") for x in nn_r2]))
print("Gradient boosting regressor:", "&".join([format(x, ".3f") for x in g_boost_r2]))


"""
######  MULTI DIMENSIONAL TESTING ALL RUNNERS ########
lin_r2 = [0]
nn_r2 = [r2_score(X_test[col_list[3:]], networks[0].predict(X_test[col_list[:3]]))]
bq_nn_r2 = [r2_score(X_test[col_list[3:]], bq_networks[0].predict(X_test[col_list[:3]]))]

for i in range(4, len(col_list)):
    #print(col_list[i])
    y_true = X_test.iloc[:, i:].values
    lin_pred = clf.predict(X_test.iloc[:,3:i].values)
    #print('Linear model', r2_score(y_true, lin_pred))
    #print('Linear model FINISH TIME', r2_score(y_true[:,-1], lin_pred[:,-1]))
    lin_r2.append(r2_score(y_true, lin_pred))
    

    train_cols = col_list[:i]
    pred_cols = col_list[i:]
    nn_pred = networks[i-3].predict(X_test[train_cols])
    bq_nn_pred = bq_networks[i-3].predict(X_test[train_cols])
    
    if i == len(col_list) - 1:
        nn_pred = nn_pred.reshape(-1,1)
        bq_nn_pred =  bq_nn_pred.reshape(-1,1)
        
    
    #print(col_list[i], 'neural model', networks[i-3].score(X_test[train_cols], X_test[pred_cols]))
    #print('neural model FINISH TIME', r2_score(y_true[:,-1], nn_pred[:,-1]))
    nn_r2.append(r2_score(y_true[:,-1], nn_pred[:,-1]))
    #print('BQ neural model FINISH TIME', r2_score(y_true[:,-1], bq_nn_pred[:,-1]))
    bq_nn_r2.append(r2_score(y_true[:,-1], bq_nn_pred[:,-1]))
    #print('g boost model FINISH TIME', r2_score(y_true[:,-1], g_boost_pred[:,-1]))
    g_boost_r2.append(r2_score(y_true[:,-1], g_boost_pred[:,-1]))
    #print('BQ g boost model FINISH TIME', r2_score(y_true[:,-1], bq_g_boost_pred[:,-1]))
    bq_g_boost_r2.append(r2_score(y_true[:,-1], bq_g_boost_pred[:,-1]))

x_ticks = ['start', '5k', '10k', '15k', '20k', 'Half', '25k', '30k', '35k', '40k']
plt.title('R2_scores for finish time - All runners')
plt.plot(range(10), lin_r2)
plt.plot(range(10), nn_r2)
plt.plot(range(10), bq_nn_r2)
plt.plot(range(10), g_boost_r2)
plt.plot(range(10), bq_g_boost_r2)
plt.xticks(range(10), x_ticks)
plt.legend(['lin', 'nn', 'bq_nn', 'g_boost', 'bq_g_boost'])
plt.savefig('./results/general_finish.png')
plt.show()
"""
