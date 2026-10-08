# Food Adulteration ML Project 

Predicts **severity**, **health risk** and **action taken** for a food adulteration
report using product, brand, category, adulterant, detection method and date.

## Structure
```
food_adulteration_ml/
├── app.py              # Flask app (UI + /predict JSON API)
├── train.py            # preprocessing, model comparison, evaluation, saving
├── data/               # food_adulteration_data.csv
├── models/             # generated: *.joblib + metadata.json
├── static/             # generated: confusion matrices + report charts
├── notebook/           # Food_Adulteration_ML.ipynb (run in Google Colab)
├── report/             # project report (PDF and Word)
├── tools/              # scripts that generated the report and notebook
├── templates/          # base, index (predict form), dashboard
└── requirements.txt
```

## Run
```bash
pip install -r requirements.txt
python train.py     # trains + evaluates, writes models/
python app.py       # open http://127.0.0.1:5000
```

## API
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
  "product_name":"Milk","brand":"BrandA","category":"Dairy","adulterant":"Melamine",
  "detection_method":"Spectroscopy","detection_date":"2024-05-10"}'
```
Also: `GET /api/metrics`, `GET /dashboard`.

## Method
- One-hot encoding for categorical fields, plus month and day-of-week from the date.
- Per target: Baseline, Logistic Regression, Random Forest, Gradient Boosting compared with
  stratified 5-fold CV; best is scored on a 20% held-out test set, then refit on all data.

## Important finding
All models score at about the majority-class baseline (about 33-36% for 3 classes, about 25% for 4).
The dataset appears to be randomly generated: labels are close to uniform, product/category pairs are
inconsistent (e.g. "Butter" in "Meat"), and severity is unrelated to health risk. No algorithm can find
signal that isn't there. To get a useful model you need real data with genuine relationships
(e.g. lab measurements, adulterant concentration, region, inspection history).

## Google Colab
Upload `notebook/Food_Adulteration_ML.ipynb` to Colab, choose Runtime > Run all, and upload
`data/food_adulteration_data.csv` when asked. The notebook trains the models and runs the Flask app inside Colab.
