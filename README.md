# House Price Prediction – King County, WA

**Machine Learning – Assignment 3:** Applying, Evaluating and Comparing Multiple ML Algorithms on a Large Real-World Dataset

| | |
|---|---|
| **Student** | Shaiza Butt (FA24-BED-066) |
| **Instructor** | Dr. Rao Muhammad Adeel Nawab |
| **GitHub** | https://github.com/ShaizaButt |
| **Repository** | https://github.com/ShaizaButt/house-price-prediction |
| **Live app** | https://house-price-prediction-shaiza-butt.streamlit.app/ |

## What this project does

It predicts the **sale price (USD) of a house** in King County, Washington (the Seattle area) from 18 features such as living area, number of bedrooms and bathrooms, construction grade, year built, ZIP code and location.

Five regression algorithms are compared with **10-fold cross validation** on the **full dataset** (21,613 house sales; 21,597 remain after cleaning), using **RMSE** as the evaluation measure. The best algorithm is retrained on all the data, saved as `model.pkl`, and used by a Streamlit web app.

## Results (10-fold cross validation)

| Rank | Algorithm | Mean RMSE | Std RMSE |
|:---:|---|---:|---:|
| 1 | **XGBoost Regressor** | **$116,745** | $10,541 |
| 2 | Extra Trees | $125,050 | $10,933 |
| 3 | Lasso Regression | $161,347 | $10,869 |
| 4 | Ridge Regression | $161,356 | $10,851 |
| 5 | AdaBoost Regressor | $367,798 | $23,925 |

The average house price in the data is $540,136, so the best model's typical error is about 21.6% of the average price. The saved `model.pkl` is the XGBoost pipeline (pre-processing + algorithm), 0.74 MB in size.

## Files

| File | Description |
|---|---|
| `House_price_prediction_kfold.ipynb` | Full notebook: data loading, cleaning, pipeline, 5 algorithms, 10-fold cross validation, comparison, final model, application and feedback phases |
| `model.pkl` | The saved best model (complete scikit-learn pipeline, saved with joblib) |
| `app.py` | Streamlit web app that loads `model.pkl` and predicts a price from a form |
| `requirements.txt` | Library versions used for training and for the app |
| `kc_house_data.csv` | Dataset (also used by the app for valid input ranges and typical values) |
| `README.md` | This file |

## Run the notebook (Google Colab)

1. Open https://colab.research.google.com and choose **File → Upload notebook** → `House_price_prediction_kfold.ipynb`.
2. Click **Runtime → Run all**.
3. When Step 2.1 shows the **Choose Files** button, select `kc_house_data.csv`.

The 10-fold run takes roughly 10 minutes (Extra Trees is the slowest algorithm).

## Run the app on your computer

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Use Python 3.11 so that the library versions in `requirements.txt` install exactly.

## Deploy on Streamlit Community Cloud

1. Create a **public** GitHub repository named `house-price-prediction` under the account `ShaizaButt`.
2. Upload `app.py`, `model.pkl`, `requirements.txt`, `kc_house_data.csv`, `README.md` and the notebook to the repository (all files in the main branch, not in a sub-folder).
3. Open https://share.streamlit.io, sign in with GitHub and click **Create app**.
4. Select the repository `ShaizaButt/house-price-prediction`, branch `main`, main file path `app.py`.
5. Set the **App URL** to `house-price-prediction-shaiza-butt`, so that the link is exactly the one written in the first cell of the notebook.
6. Under **Advanced settings** choose **Python 3.11**, then click **Deploy**.

Keep the library versions in `requirements.txt`: `model.pkl` was trained with them, and a different scikit-learn or XGBoost version may fail to load it.

## Dataset

**House Sales in King County, USA** by harlfoxem on Kaggle (homes sold between May 2014 and May 2015), released under **CC0 1.0 (Public Domain)**: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
