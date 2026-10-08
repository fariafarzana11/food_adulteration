import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

M = json.load(open("models/metadata.json"))
S, E = M["summary"], M["eda"]
GREEN = colors.HexColor("#2f7d4f")

# ---------- charts ----------
def bar(d, title, fn, color="#2f7d4f"):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(list(d.keys()), list(d.values()), color=color)
    ax.set_title(title, fontsize=10)
    plt.xticks(rotation=25, ha="right", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout(); fig.savefig(fn, dpi=130); plt.close(fig)

bar(E["by_adulterant"], "Records by adulterant", "static/r_adulterant.png")
bar(E["by_severity"], "Records by severity", "static/r_severity.png", "#b45309")
bar(E["by_health_risk"], "Records by health risk", "static/r_risk.png", "#2563eb")
bar(E["by_category"], "Records by category", "static/r_category.png", "#7c3aed")

fig, ax = plt.subplots(figsize=(6.2, 3.2))
names = list(S.keys())
x = range(len(names))
ax.bar([i - 0.2 for i in x], [S[n]["test_accuracy"] for n in names], 0.4, label="Best model", color="#2f7d4f")
ax.bar([i + 0.2 for i in x], [S[n]["baseline_accuracy"] for n in names], 0.4, label="Majority baseline", color="#9ca3af")
ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0, 0.5); ax.set_ylabel("Test accuracy"); ax.legend(fontsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout(); fig.savefig("static/r_compare.png", dpi=130); plt.close(fig)

# ---------- styles ----------
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=7)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, textColor=GREEN, spaceBefore=12, spaceAfter=8)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=colors.HexColor("#1f2933"), spaceBefore=8, spaceAfter=5)
cap = ParagraphStyle("c", parent=ss["Normal"], fontSize=8.5, alignment=TA_CENTER, textColor=colors.grey, spaceAfter=10)
bul = ParagraphStyle("bl", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=3)
title = ParagraphStyle("t", parent=ss["Title"], fontSize=24, textColor=GREEN, leading=30)
sub = ParagraphStyle("s", parent=ss["Normal"], fontSize=13, alignment=TA_CENTER, textColor=colors.HexColor("#4b5563"), leading=18)

cell = ParagraphStyle("cell", parent=ss["Normal"], fontSize=9, leading=11)
def P(t): return Paragraph(t, body)
def B(items): return [Paragraph(i, bul, bulletText="\u2022") for i in items]

def table(data, widths=None, hdr=True):
    t = Table(data, colWidths=widths, hAlign="CENTER")
    st = [("FONTSIZE", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]
    if hdr:
        st += [("BACKGROUND", (0, 0), (-1, 0), GREEN), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")])]
    t.setStyle(TableStyle(st))
    return t

def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
    c.drawString(2 * cm, 1.2 * cm, "Food Adulteration Detection using Machine Learning")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {d.page}")
    c.restoreState()

doc = SimpleDocTemplate("/mnt/user-data/outputs/Food_Adulteration_ML_Report.pdf", pagesize=A4,
                        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                        title="Food Adulteration Analysis using Machine Learning", author="")
st = []

# ---------- cover ----------
st += [Spacer(1, 5 * cm), Paragraph("Food Adulteration Analysis<br/>using Machine Learning", title), Spacer(1, 0.6 * cm),
       Paragraph("Classification of severity, health risk and regulatory action<br/>with a Flask web application", sub),
       Spacer(1, 1.5 * cm),
       Paragraph(f"Dataset: {E['n_records']} adulteration records &nbsp;|&nbsp; {E['date_range'][0]} to {E['date_range'][1]}", sub),
       Spacer(1, 0.4 * cm), Paragraph("Tools: Python, scikit-learn, pandas, Flask", sub), PageBreak()]

# ---------- abstract ----------
st += [Paragraph("Abstract", h1),
       P("This project builds a machine learning pipeline and a Flask web application on a dataset of 1,000 food "
         "adulteration reports. Three classification tasks are studied: predicting the <b>severity</b> of a case, its "
         "<b>health risk</b>, and the <b>action taken</b> by authorities, using product, brand, category, adulterant, "
         "detection method and detection date as inputs. Logistic Regression, Random Forest and Gradient Boosting were "
         "compared against a majority-class baseline using stratified 5-fold cross-validation and a held-out test set."),
       P("<b>The main finding is negative:</b> none of the models beat the baseline in any meaningful way "
         "(test accuracy 29.5%, 37.0% and 20.5% against baselines of 35.5%, 36.0% and 26.5%). Chi-square tests show no "
         "statistically significant relationship between any input feature and any target, and the data contains "
         "logically inconsistent records. The dataset therefore appears to be randomly generated and contains no "
         "learnable signal. The report documents the full pipeline, explains this result and recommends what data "
         "would be needed for a useful system."),
       Paragraph("1. Introduction", h1),
       P("Food adulteration, the addition of inferior or harmful substances to food, is a public health and economic "
         "concern. Regulators record each detected case together with its severity, the health risk and the action "
         "taken. A model that could anticipate these outcomes from basic case details could help prioritise inspections "
         "and responses."),
       P("The objectives of this project are:"),
       *B(["to explore and clean the provided adulteration dataset;",
           "to train and compare several classifiers for three target variables;",
           "to evaluate them honestly against a naive baseline;",
           "to deploy the trained models in a Flask web application with a prediction form, a JSON API and a dashboard."]),
       PageBreak(),
       Paragraph("2. Dataset", h1),
       P(f"The dataset has {E['n_records']} rows and 10 columns, with no missing values or duplicate IDs. Dates span "
         f"{E['date_range'][0]} to {E['date_range'][1]}."),
       table([["Column", "Type", "Description / values"],
              ["adulteration_id", "ID", "Unique record number (dropped from modelling)"],
              ["product_name", "Categorical", "10 products (Milk, Honey, Wine, Beef ...)"],
              ["brand", "Categorical", "BrandA to BrandE"],
              ["category", "Categorical", "Meat, Dairy, Bakery, Beverages, Condiments"],
              ["adulterant", "Categorical", "Chalk, Water, Melamine, Coloring agents, Artificial sweeteners"],
              ["detection_date", "Date", "Converted to month and day-of-week"],
              ["detection_method", "Categorical", "Spectroscopy, Chemical, Microbiological, Sensory"],
              ["severity", "Target", "Minor / Moderate / Severe"],
              ["health_risk", "Target", "Low / Medium / High"],
              ["action_taken", "Target", "Investigation, Recall, Fine, Warning"]],
             [3.6 * cm, 2.8 * cm, 10.4 * cm]),
       Paragraph("Table 1: Dataset columns", cap)]

# ---------- EDA ----------
st += [Paragraph("3. Exploratory Data Analysis", h1),
       P("Every categorical variable is spread almost evenly across its classes. No class dominates, which means the "
         "majority-class baseline accuracy is only 35.5% for severity, 36.0% for health risk and 26.5% for action taken."),
       Table([[Image("static/r_adulterant.png", 7.0 * cm, 4.2 * cm), Image("static/r_category.png", 7.0 * cm, 4.2 * cm)],
              [Image("static/r_severity.png", 7.0 * cm, 4.2 * cm), Image("static/r_risk.png", 7.0 * cm, 4.2 * cm)]]),
       Paragraph("Figure 1: Class distributions of key variables", cap),
       Paragraph("3.1 Signs of synthetic data", h2),
       P("Three checks suggest the data was generated by random sampling rather than collected from real cases:"),
       *B(["<b>Chi-square tests of independence</b> between each of the five input features and each of the three "
           "targets gave p-values between 0.29 and 0.98. None is below 0.05, so no feature is statistically related to any outcome.",
           "<b>Severity and health risk are unrelated</b> (p = 0.59). In real records, severe cases would be expected to carry "
           "higher health risk.",
           "<b>Inconsistent records.</b> Products appear under implausible categories, for example Butter under Meat, "
           "Wine under Dairy and Bread under Beverages. Product and category are close to independent."]),
       Paragraph("4. Methodology", h1),
       Paragraph("4.1 Preprocessing", h2),
       *B(["Dates were parsed and converted into <i>month</i> and <i>day_of_week</i>.",
           "The ID column was removed.",
           "Categorical features were one-hot encoded inside a scikit-learn Pipeline (unknown categories ignored).",
           "Each target was modelled separately with an 80/20 stratified train/test split (random_state = 42)."]),
       Paragraph("4.2 Models", h2),
       *B(["<b>Baseline:</b> always predicts the most frequent class.",
           "<b>Logistic Regression:</b> linear model, max_iter = 1000.",
           "<b>Random Forest:</b> 200 trees, min_samples_leaf = 5.",
           "<b>Gradient Boosting:</b> 100 trees, max_depth = 2."]),
       Paragraph("4.3 Evaluation", h2),
       P("Models were compared with stratified 5-fold cross-validation on the training set. The best non-baseline model per "
         "target was scored once on the held-out test set using accuracy and macro-F1, then refit on all data for deployment. "
         "A model is only useful if it clearly outperforms the baseline."),
       PageBreak()]

# ---------- results ----------
rows = [["Model"] + [t.replace("_", " ").title() for t in S]]
for m in S["severity"]["cv_scores"]:
    rows.append([m] + [f"{S[t]['cv_scores'][m]:.3f}" for t in S])
st += [Paragraph("5. Results", h1), Paragraph("5.1 Cross-validation accuracy", h2),
       table(rows, [6 * cm, 3.4 * cm, 3.4 * cm, 3.4 * cm]),
       Paragraph("Table 2: Mean 5-fold CV accuracy on the training set", cap),
       P("In cross-validation, no model beats the baseline for severity or health risk. For action taken, Random Forest "
         "reaches 0.280 against 0.265, a difference well within the fold-to-fold variation (about 0.03)."),
       Paragraph("5.2 Held-out test performance", h2),
       table([["Target", "Best model", "Accuracy", "Macro-F1", "Baseline acc."]] +
             [[t.replace("_", " ").title(), S[t]["best_model"], f"{S[t]['test_accuracy']:.3f}",
               f"{S[t]['test_macro_f1']:.3f}", f"{S[t]['baseline_accuracy']:.3f}"] for t in S],
             [3.6 * cm, 3.6 * cm, 3 * cm, 3 * cm, 3.4 * cm]),
       Paragraph("Table 3: Test-set results", cap),
       Image("static/r_compare.png", 11.5 * cm, 5.9 * cm),
       Paragraph("Figure 2: Best model versus majority baseline", cap),
       P("Severity and action taken score below the baseline, and health risk is only 1 percentage point above it. With "
         "200 test rows, one percentage point is two predictions, so this is noise. Accuracy near 33% for three classes "
         "and near 25% for four classes is exactly what random guessing produces."),
       PageBreak(),
       Paragraph("5.3 Confusion matrices", h2),
       Table([[Image("static/cm_severity.png", 8.2 * cm, 6.6 * cm), Image("static/cm_health_risk.png", 8.2 * cm, 6.6 * cm)],
              [Image("static/cm_action_taken.png", 8.2 * cm, 6.6 * cm), ""]]),
       Paragraph("Figure 3: Confusion matrices on the test set (left to right, top to bottom: severity, health risk, action taken)", cap),
       P("Predictions are scattered across all classes with no strong diagonal, which confirms that the models cannot "
         "separate the classes."),
       Paragraph("6. Web Application", h1),
       P("The trained pipelines are served by a Flask application:"),
       table([["Route", "Method", "Purpose"],
              ["/", "GET", Paragraph("Form to enter case details and view predictions with class probabilities", cell)],
              ["/predict", "POST", Paragraph("JSON API returning prediction, probabilities and model accuracy for all three targets", cell)],
              ["/dashboard", "GET", "Dataset summary, model comparison table and confusion matrices"],
              ["/api/metrics", "GET", "Evaluation metrics in JSON"]],
             [3.2 * cm, 2.2 * cm, 11.4 * cm]),
       Paragraph("Table 4: Application routes", cap),
       P("The app validates every input (allowed category values and date format) and returns HTTP 400 for invalid requests. "
         "Because the models are not reliable, the prediction page shows a visible notice and always displays each model's "
         "accuracy next to the baseline so that users do not mistake the output for real guidance."),
       Paragraph("7. Discussion and Limitations", h1),
       *B(["<b>No learnable signal.</b> The negative result comes from the data, not from the algorithms or the code. "
           "Tuning hyperparameters or trying deeper models would not help.",
           "<b>Limited features.</b> Only descriptive fields are available. Real predictors such as adulterant concentration, "
           "lab measurements, region, supplier history or inspector notes are missing.",
           "<b>Small sample and short period.</b> 1,000 rows over six months cannot capture seasonality or rare events.",
           "<b>Label definitions.</b> It is unclear how severity and risk were assigned, and the dataset shows no link between them."]),
       Paragraph("8. Conclusion and Future Work", h1),
       P("A complete, reproducible pipeline and web application were built, and the evaluation was done honestly against a "
         "baseline. The result is that this dataset cannot support predictive modelling: all models perform at chance level and "
         "statistical tests find no relationships. The system is best seen as a working template that can be retrained on "
         "better data by replacing the CSV."),
       P("Recommended next steps:"),
       *B(["collect real records with measurable features (adulterant concentration, test readings, region, date of production);",
           "verify that product and category values are consistent;",
           "define severity and risk labels with clear rules so they relate to each other;",
           "re-run the pipeline, add feature importance analysis and consider cost-sensitive metrics, since missing a high-risk case is worse than a false alarm."]),
       Paragraph("Appendix: How to run", h1),
       P("<font face='Courier' size='9'>pip install -r requirements.txt<br/>python train.py<br/>python app.py</font>"),
       P("Then open http://127.0.0.1:5000 in a browser. On Google Colab, the app is run in a background thread and displayed "
         "with <font face='Courier'>output.serve_kernel_port_as_iframe(5000)</font>.")]

doc.build(st, onFirstPage=lambda c, d: None, onLaterPages=footer)
print("done")
