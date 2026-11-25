# Disease Risk Screening



https://github.com/user-attachments/assets/88aa957e-006f-4d15-a415-918b4a8a5a82



A small app that flags possible heart disease, diabetes and breast cancer from patient data. I built it during my internship at [CodeAlpha/Machine Learning].

**Try it:** (https://codealpha---disease-risk-screening-hhwtqcazh7qn7f5fx69asy.streamlit.app/) (if it's asleep, click "wake up" and give it a minute)

## Why I built it
My task was to predict diseases from medical data using SVM, Logistic Regression, Random Forest and XGBoost. I wanted the result to be something you can actually click around in, not just a notebook with numbers.

## What you can do in the app
- Pick a condition and enter patient values, or load a real test patient.
- Drag the threshold and see sick patients caught vs healthy people falsely flagged.
- See which inputs pushed the score up or down.
- Compare all four algorithms.

## Problems I hit and what I did
- **Diabetes data had zeros that weren't real.** A glucose of 0 means "not measured". I turned them into missing values and handled them properly. Insulin was almost half missing.
- **Easy to cheat by accident.** Filling missing values before splitting leaks test data into training. I did all preprocessing inside a pipeline.
- **The four algorithms scored almost the same.** Differences were smaller than the noise, so I picked Logistic Regression because it's simple and explainable.
- **The default cutoff missed too many sick people.** On diabetes it missed 17 of 54 test patients. Lowering the cutoff cut that to 6, but 12 more healthy people got flagged. That trade-off is the slider in the app.
- **XGBoost looked best on one test table.** It wasn't. Its ranking ability (AUC) was identical, it just used a different cutoff. I didn't switch models.
- **My Colab demo link kept dying.** I moved the app to GitHub and Streamlit Cloud so the link stays up.

## Results (held-out test set)
| Dataset | Test patients | AUC | Sick caught |
|---|---|---|---|
| Heart | 61 | 0.965 | 27 of 28 |
| Diabetes | 154 | 0.816 | 48 of 54 |
| Cancer | 114 | 0.995 | 40 of 42 |

Heart's cross-validated AUC was about 0.90, so 0.965 is probably a lucky small test set. On cancer I aimed for higher recall and missed 2 tumors.

## Tech stack
- **Python** on Google Colab (GPU Tesla 4)
- **pandas, NumPy** for data
- **scikit-learn** for pipelines, SVM, Logistic Regression, Random Forest, cross-validation and tuning
- **XGBoost** for gradient boosting
- **Plotly** for charts
- **Streamlit** for the app, hosted on Streamlit Community Cloud
- **Data:** UCI Heart Disease, Pima Indians Diabetes, Breast Cancer Wisconsin

## Limits
Small datasets, so results are shaky. Each model only fits the group it was trained on (the diabetes model is Pima women only). Scores are not true probabilities. This is a learning project, not medical advice.

## Run it yourself
