# Insurance Premium Predictor

A simple end-to-end ML demo: a linear regression model trained on insurance
policy data, served through a Flask API, with a static HTML front end for
entering policy details and getting an estimated annual premium.

## Project structure

```
.
├── insurance_linear_regression_raw_2000.csv   # raw training data (2000 rows)
├── Making_model1.ipynb                        # notebook: trains the model, saves model.pkl
├── model.pkl                                  # trained sklearn Pipeline (preprocessing + regression)
├── app.py                                     # Flask app: serves the UI and the /predict API
├── index.html                                 # front end (served by app.py at /)
└── README.md
```

## Dataset

`insurance_linear_regression_raw_2000.csv` has one row per customer/policy,
with these columns:

| Column | Type | Description |
|---|---|---|
| `Customer_ID` | id | Not used as a feature |
| `Policy_Type` | categorical | Auto, Health, Home, Life, Travel |
| `Customer_Segment` | categorical | Mass, Premium, Affluent, Corporate |
| `Customer_Age` | numeric | Customer's age |
| `Annual_Income` | numeric | Customer's annual income |
| `Claim_History_Count` | numeric | Number of past claims |
| `Previous_Claim_Amount` | numeric | Total amount of past claims |
| `Policy_Tenure_Years` | numeric | Years the policy has been held |
| `Risk_Score` | numeric | Underwriting risk score |
| `Insured_Value` | numeric | Value of the insured asset |
| `Annual_Insurance_Premium` | numeric | **Target** — what the model predicts |

**Note on data quality:** this is raw, unclean data — it contains missing
values, inconsistent category spellings/casing (e.g. `Auto` vs `AUTO` vs
`auto`), and some outliers/negative values. The current notebook only drops
rows with missing values in the used columns; it does **not** standardize
category text or remove outliers. As a result, the model's test R² is
around 0.58 — usable for a demo, but not something to trust for real
underwriting decisions. See "Improving the model" below if you want to fix
this.

## How it works

1. **`Making_model1.ipynb`** loads the CSV, splits it into train/test sets,
   and fits an sklearn `Pipeline` with:
   - `ColumnTransformer`: numeric columns pass through unchanged, categorical
     columns (`Policy_Type`, `Customer_Segment`) go through `OneHotEncoder`.
   - `LinearRegression` on top.

   It prints MAE / RMSE / R² on the test set, then saves the whole pipeline
   to `model.pkl` with `pickle`.

2. **`app.py`** loads `model.pkl` and exposes:
   - `GET /` — serves `index.html`.
   - `POST /predict` — accepts a JSON body with the 9 feature fields, builds
     a one-row DataFrame with the same column names used in training, runs
     `model.predict(...)`, and returns `{"success": true, "prediction": <number>}`.

3. **`index.html`** is a static form. On submit, it POSTs the field values
   as JSON to `/predict` and displays the returned premium, along with a
   summary of what was entered.

## Running it locally

Requirements: Python 3.9+, and:

```bash
pip install flask pandas numpy scikit-learn
```

1. (Optional) Retrain the model — only needed if you change the data or
   feature set:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace Making_model1.ipynb
   ```
   This regenerates `model.pkl`.

2. Start the server:
   ```bash
   python app.py
   ```

3. Open **http://127.0.0.1:5000** in a browser, fill in the form, and click
   **Calculate premium**.

## API reference

`POST /predict`

Request body:
```json
{
  "policy_type": "Auto",
  "customer_segment": "Premium",
  "customer_age": 35,
  "annual_income": 900000,
  "claim_history_count": 1,
  "previous_claim_amount": 50000,
  "policy_tenure_years": 5,
  "risk_score": 45,
  "insured_value": 800000
}
```

Success response:
```json
{ "success": true, "prediction": 82485.59 }
```

Error response (missing field, bad number, etc.):
```json
{ "success": false, "error": "..." }
```

## Improving the model

If you want a more trustworthy model, the main next step is cleaning the
raw CSV before training:
- Standardize category text (trim whitespace, fix casing, fix typos like
  `Corporrate` → `Corporate`).
- Handle or remove unrealistic values (negative premiums/insured values,
  implausible ages).
- Consider capping or removing extreme outliers before fitting.

These changes would happen in `Making_model1.ipynb`, upstream of the
`train_test_split` call — the rest of the pipeline (encoding, regression,
Flask API, front end) would not need to change.
