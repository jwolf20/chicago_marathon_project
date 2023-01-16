# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:04:08 2018

@author: J Wolf
"""
import glob

import pandas as pd


def convert_time(time_string):
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


def age_index(age):
    if age == "M-15" or age == "W-15":
        return 0
    else:
        return [
            0,
            "16-19",
            "20-24",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            "60-64",
            "65-69",
            "70-74",
            "75-79",
            "80+",
        ].index(age)


def sex_index(sex):
    return [0, 1][sex == "female"]


data_file_paths = glob.glob("./data_files/*_chicago.csv")
frames = [pd.read_csv(path, low_memory=False) for path in data_file_paths]

drop_list = ["age", "bib", "city", "initials"]

splits = ["5k", "10k", "15k", "20k", "Half", "25k", "30k", "35k", "40k", "Finish"]
split_fields = ["time of day", "time", "diff", "min/mile", "miles/h"]
split_times = [
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

for df in frames:
    df.drop(drop_list, axis=1, inplace=True)
    df.dropna(inplace=True)

data = pd.concat(frames)

data = data[data["5k split time"] != "-"]
data["age group"] = data["age group"].map(age_index)
data["sex"] = data["sex"].map(sex_index)
data["starting time"] = data["starting time"].map(convert_time)
data["overall time"] = data["overall time"].map(convert_time)
for split in splits:
    for field in split_fields:
        if field != "miles/h":
            data[split + " split " + field] = data[split + " split " + field].map(
                convert_time
            )
        else:
            data[split + " split " + field] = pd.to_numeric(
                data[split + " split " + field]
            )

data = data[data["starting time"] >= 27000]  # NOTE: Remove people that started early
data.to_csv("./data_files/clean_chicago_data.csv")

# Construct the set of BQ athletes
males = data[data["sex"] == 0]
females = data[data["sex"] == 1]

BQ_male_cutoffs = [
    "03:05:00",
    "03:10:00",
    "03:15:00",
    "03:25:00",
    "03:30:00",
    "03:40:00",
    "03:55:00",
    "04:10:00",
    "04:25:00",
    "04:40:00",
    "04:55:00",
]

BQ_female_cutoffs = [
    "03:35:00",
    "03:40:00",
    "03:45:00",
    "03:55:00",
    "04:00:00",
    "04:10:00",
    "04:25:00",
    "04:40:00",
    "04:55:00",
    "05:10:00",
    "05:25:00",
]

BQ_male_cutoffs = [convert_time(t) for t in BQ_male_cutoffs]
BQ_female_cutoffs = [convert_time(t) for t in BQ_female_cutoffs]

BQ_data = males[
    (males["age group"] <= 4) & (males["Finish split time"] <= BQ_male_cutoffs[0])
]
for i in range(1, 11):
    quals = males[
        (males["age group"] == i + 4)
        & (males["Finish split time"] <= BQ_male_cutoffs[i])
    ]
    BQ_data = pd.concat([BQ_data, quals])

quals = females[
    (females["age group"] <= 4) & (females["Finish split time"] <= BQ_female_cutoffs[0])
]
BQ_data = pd.concat([BQ_data, quals])
for i in range(1, 11):
    quals = females[
        (females["age group"] == i + 4)
        & (females["Finish split time"] <= BQ_female_cutoffs[i])
    ]
    BQ_data = pd.concat([BQ_data, quals])

BQ_data.to_csv("./data_files/clean_chicago_bq_data.csv")
