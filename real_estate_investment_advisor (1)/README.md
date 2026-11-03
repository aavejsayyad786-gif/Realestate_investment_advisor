# Real Estate Investment Advisor

A Streamlit demo for investment classification and five-year price estimation.

## Important
The included `data/demo_housing_data.csv` is **synthetic demonstration data**. It is not the assignment's real `india_housing_prices.csv`. Replace it with the assigned dataset before treating metrics or predictions as real.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, choose the repository and `app.py`.
3. Deploy. Include `requirements.txt`.
4. If using the real dataset, commit it only if you have permission and it contains no sensitive data.

## Method notes
- Investment class follows the brief's price-at-or-below-median rule.
- Five-year target uses the brief's example fixed 8% annual growth assumption.
- Since these labels are rule-derived, the model learns those rules; it does not establish real-world investment performance.
- MLflow logs model parameters, metrics, and artifacts.
