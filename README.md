# Chicago Marathon Project

This project contains several models for predicting the future splits of runners in the Chicago marathon based on prior in race splits.

## About

This repository contains the code used for a project I did in Spring 2018.

The project started based on the observation that the Chicago marathon used a rather simple approach for tracking runnings within the race and attempting to predict future splits. That approach was simply a linear extrapolation of the pace from the most recent split.

The goal of this project was to build a model that generated more accurate predictions compared to such a simplistic approach.

## Scripts

- webcrawler_chicago_mar/
  - The folder contains the code used for scraping the original training data sets from the web.
  - NOTE: Some of the links used in the scrapping may no longer be active (they were valid at the time this project was written).
- basic_linear_model.py

  - This script contains the code for the basic linear model which serves as the baseline model for this project. The purpose of this model is that it will generate predictions for all remaining splits by linear extrapolation of the pace of the most recent split.

- data_cleanup.py

  - This script is used to clean and combine the marathon results gather from the web scrapping into files that are more easily processed for model training.

  - One of the main steps in the cleaning process is to remove any data points which are missing splits.

  - Result of executing this script is the creation of the following files:
    - ./data_files/clean_chicago_data.csv
    - ./data_files/clean_chicago_bq_data.csv

- model_training.py
  - This scripts runs the model training and stores the pickled models in the `/models/` folder.
  - The different models used in this project are as follows:
    - lin: The baseline linear extrapolation model.
    - nn: A MLPRegressor model.
    - g_boost: A GradientBoostingRegressor model.
    - bq_nn: Similar to nn; except only trained on the data for BQ runners.
    - bq_g_boost: Similar to g_boost; except only trained on the data for BQ runners.
  - This script also runs the comparison of the different models and stores some graphics related to the results in the `/results/` folder.

## Notes

- The models are defined and named based on the number of splits that they have information about (i.e., nn2 is the model that generates predications based on the 5k and 10k splits; while nn3 additionally includes the 15k split time).
- The results show that the models are a clear improvement compared to the baseline model. Especially among the splits in the first half of the race where the baseline model has an r2_score below 0.8, while both models show r2_scores near 0.9 (for predicting finish times based on early splits).
- This also shows that the BQ standards represent a dividing line among the population of runners. Moreover this dividing line is witnessed by the approaches here as the models trained on BQ data vastly outperform the general models on that dataset. While the BQ models do not perform nearly as well when applied to the broader class of all marathon runners.
- The data files containing the results of all the splits for finishers of the Chicago Marathon from 2014-2017 are not included in this repository.
  - Instead I have included the code used for the web scrapper I used to pull the data. This web scrapper was built using the `scrappy` package. That code is located in the `webcrawler_chicago_mar` folder of this repository.
