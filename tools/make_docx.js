const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, Footer,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, TabStopType,
} = require("docx");

const M = JSON.parse(fs.readFileSync("models/metadata.json", "utf8"));
const S = M.summary, E = M.eda;
const GREEN = "2F7D4F";
const W = 9638; // content width (A4, 2cm margins)

const run = (t, o = {}) => new TextRun({ text: t, font: "Calibri", size: 22, ...o });
// paragraph from segments: strings or [text, opts]
const para = (segs, o = {}) => new Paragraph({
  spacing: { after: 120, line: 300 }, alignment: AlignmentType.JUSTIFIED, ...o,
  children: (Array.isArray(segs) ? segs : [segs]).map(s => typeof s === "string" ? run(s) : run(s[0], s[1])),
});
const bullet = (segs) => new Paragraph({
  numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 290 },
  children: (Array.isArray(segs) ? segs : [segs]).map(s => typeof s === "string" ? run(s) : run(s[0], s[1])),
});
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 140 }, children: [new TextRun({ text: t })] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 180, after: 100 }, children: [new TextRun({ text: t })] });
const cap = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200 },
  children: [new TextRun({ text: t, font: "Calibri", size: 18, italics: true, color: "6B7280" })] });
const b = (t) => [t, { bold: true }];
const it = (t) => [t, { italics: true }];

const border = { style: BorderStyle.SINGLE, size: 4, color: "D1D5DB" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(rows, widths) {
  const total = widths.reduce((a, c) => a + c, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
    rows: rows.map((r, ri) => new TableRow({
      tableHeader: ri === 0, cantSplit: true,
      children: r.map((c, ci) => new TableCell({
        width: { size: widths[ci], type: WidthType.DXA }, borders,
        margins: { top: 70, bottom: 70, left: 110, right: 110 },
        shading: ri === 0 ? { fill: GREEN, type: ShadingType.CLEAR, color: "auto" }
          : ri % 2 === 0 ? { fill: "F3F4F6", type: ShadingType.CLEAR, color: "auto" } : undefined,
        children: [new Paragraph({ children: [new TextRun({ text: String(c), font: "Calibri", size: 19,
          bold: ri === 0, color: ri === 0 ? "FFFFFF" : "1F2933" })] })],
      })),
    })),
  });
}
const img = (file, wPx, hPx) => new ImageRun({ type: "png", data: fs.readFileSync(file),
  transformation: { width: wPx, height: hPx },
  altText: { title: file, description: file, name: file } });
const imgPara = (file, w, h) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60 }, keepNext: true, children: [img(file, w, h)] });
const imgPair = (f1, f2, w, h) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60 }, keepNext: true,
  children: [img(f1, w, h), run("  "), img(f2, w, h)] });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

const names = Object.keys(S);
const tt = (s) => s.replace("_", " ").replace(/\b\w/g, c => c.toUpperCase());
const cvRows = [["Model", ...names.map(tt)]];
for (const m of Object.keys(S.severity.cv_scores)) cvRows.push([m, ...names.map(n => S[n].cv_scores[m].toFixed(3))]);
const testRows = [["Target", "Best model", "Accuracy", "Macro-F1", "Baseline acc."],
  ...names.map(n => [tt(n), S[n].best_model, S[n].test_accuracy.toFixed(3), S[n].test_macro_f1.toFixed(3), S[n].baseline_accuracy.toFixed(3)])];

const c = [];
// ---------- cover ----------
c.push(new Paragraph({ spacing: { before: 3200 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Food Adulteration Analysis", font: "Calibri", size: 56, bold: true, color: GREEN })] }));
c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
  children: [new TextRun({ text: "using Machine Learning", font: "Calibri", size: 56, bold: true, color: GREEN })] }));
c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 700 },
  children: [new TextRun({ text: "Classification of severity, health risk and regulatory action with a Flask web application", font: "Calibri", size: 28, color: "4B5563" })] }));
c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: [new TextRun({ text: `Dataset: ${E.n_records} adulteration records  |  ${E.date_range[0]} to ${E.date_range[1]}`, font: "Calibri", size: 24, color: "4B5563" })] }));
c.push(new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Tools: Python, scikit-learn, pandas, Flask", font: "Calibri", size: 24, color: "4B5563" })] }));
c.push(pageBreak());

// ---------- abstract / intro ----------
c.push(h1("Abstract"));
c.push(para("This project builds a machine learning pipeline and a Flask web application on a dataset of 1,000 food adulteration reports. Three classification tasks are studied: predicting the severity of a case, its health risk, and the action taken by authorities, using product, brand, category, adulterant, detection method and detection date as inputs. Logistic Regression, Random Forest and Gradient Boosting were compared against a majority-class baseline using stratified 5-fold cross-validation and a held-out test set."));
c.push(para([b("The main finding is negative: "), "none of the models beat the baseline in any meaningful way (test accuracy 29.5%, 37.0% and 20.5% against baselines of 35.5%, 36.0% and 26.5%). Chi-square tests show no statistically significant relationship between any input feature and any target, and the data contains logically inconsistent records. The dataset therefore appears to be randomly generated and contains no learnable signal. This report documents the full pipeline, explains the result and recommends what data would be needed for a useful system."]));
c.push(h1("1. Introduction"));
c.push(para("Food adulteration, the addition of inferior or harmful substances to food, is a public health and economic concern. Regulators record each detected case together with its severity, the health risk and the action taken. A model that could anticipate these outcomes from basic case details could help prioritise inspections and responses."));
c.push(para("The objectives of this project are:"));
["to explore and clean the provided adulteration dataset;", "to train and compare several classifiers for three target variables;",
 "to evaluate them honestly against a naive baseline;",
 "to deploy the trained models in a Flask web application with a prediction form, a JSON API and a dashboard."].forEach(t => c.push(bullet(t)));

// ---------- dataset ----------
c.push(h1("2. Dataset"));
c.push(para(`The dataset has ${E.n_records} rows and 10 columns, with no missing values or duplicate IDs. Dates span ${E.date_range[0]} to ${E.date_range[1]}.`));
c.push(table([
  ["Column", "Type", "Description / values"],
  ["adulteration_id", "ID", "Unique record number (dropped from modelling)"],
  ["product_name", "Categorical", "10 products (Milk, Honey, Wine, Beef ...)"],
  ["brand", "Categorical", "BrandA to BrandE"],
  ["category", "Categorical", "Meat, Dairy, Bakery, Beverages, Condiments"],
  ["adulterant", "Categorical", "Chalk, Water, Melamine, Coloring agents, Artificial sweeteners"],
  ["detection_date", "Date", "Converted to month and day-of-week"],
  ["detection_method", "Categorical", "Spectroscopy, Chemical, Microbiological, Sensory"],
  ["severity", "Target", "Minor / Moderate / Severe"],
  ["health_risk", "Target", "Low / Medium / High"],
  ["action_taken", "Target", "Investigation, Recall, Fine, Warning"],
], [2100, 1700, 5838]));
c.push(cap("Table 1: Dataset columns"));

// ---------- EDA ----------
c.push(h1("3. Exploratory Data Analysis"));
c.push(para("Every categorical variable is spread almost evenly across its classes. No class dominates, which means the majority-class baseline accuracy is only 35.5% for severity, 36.0% for health risk and 26.5% for action taken."));
c.push(imgPair("static/r_adulterant.png", "static/r_category.png", 300, 180));
c.push(imgPair("static/r_severity.png", "static/r_risk.png", 300, 180));
c.push(cap("Figure 1: Class distributions of key variables"));
c.push(h2("3.1 Signs of synthetic data"));
c.push(para("Three checks suggest the data was generated by random sampling rather than collected from real cases:"));
c.push(bullet([b("Chi-square tests of independence "), "between each of the five input features and each of the three targets gave p-values between 0.29 and 0.98. None is below 0.05, so no feature is statistically related to any outcome."]));
c.push(bullet([b("Severity and health risk are unrelated "), "(p = 0.59). In real records, severe cases would be expected to carry higher health risk."]));
c.push(bullet([b("Inconsistent records. "), "Products appear under implausible categories, for example Butter under Meat, Wine under Dairy and Bread under Beverages. Product and category are close to independent."]));

// ---------- method ----------
c.push(h1("4. Methodology"));
c.push(h2("4.1 Preprocessing"));
["Dates were parsed and converted into month and day_of_week.", "The ID column was removed.",
 "Categorical features were one-hot encoded inside a scikit-learn Pipeline (unknown categories ignored).",
 "Each target was modelled separately with an 80/20 stratified train/test split (random_state = 42)."].forEach(t => c.push(bullet(t)));
c.push(h2("4.2 Models"));
c.push(bullet([b("Baseline: "), "always predicts the most frequent class."]));
c.push(bullet([b("Logistic Regression: "), "linear model, max_iter = 1000."]));
c.push(bullet([b("Random Forest: "), "200 trees, min_samples_leaf = 5."]));
c.push(bullet([b("Gradient Boosting: "), "100 trees, max_depth = 2."]));
c.push(h2("4.3 Evaluation"));
c.push(para("Models were compared with stratified 5-fold cross-validation on the training set. The best non-baseline model per target was scored once on the held-out test set using accuracy and macro-F1, then refit on all data for deployment. A model is only useful if it clearly outperforms the baseline."));

// ---------- results ----------
c.push(h1("5. Results"));
c.push(h2("5.1 Cross-validation accuracy"));
c.push(table(cvRows, [3338, 2100, 2100, 2100]));
c.push(cap("Table 2: Mean 5-fold CV accuracy on the training set"));
c.push(para("In cross-validation, no model beats the baseline for severity or health risk. For action taken, Random Forest reaches 0.280 against 0.265, a difference well within the fold-to-fold variation (about 0.03)."));
c.push(h2("5.2 Held-out test performance"));
c.push(table(testRows, [1900, 2200, 1800, 1800, 1938]));
c.push(cap("Table 3: Test-set results"));
c.push(imgPara("static/r_compare.png", 440, 226));
c.push(cap("Figure 2: Best model versus majority baseline"));
c.push(para("Severity and action taken score below the baseline, and health risk is only 1 percentage point above it. With 200 test rows, one percentage point is two predictions, so this is noise. Accuracy near 33% for three classes and near 25% for four classes is exactly what random guessing produces."));
c.push(h2("5.3 Confusion matrices"));
c.push(imgPair("static/cm_severity.png", "static/cm_health_risk.png", 290, 232));
c.push(imgPara("static/cm_action_taken.png", 290, 232));
c.push(cap("Figure 3: Confusion matrices on the test set (severity, health risk, action taken)"));
c.push(para("Predictions are scattered across all classes with no strong diagonal, which confirms that the models cannot separate the classes."));

// ---------- web app ----------
c.push(h1("6. Web Application"));
c.push(para("The trained pipelines are served by a Flask application:"));
c.push(table([
  ["Route", "Method", "Purpose"],
  ["/", "GET", "Form to enter case details and view predictions with class probabilities"],
  ["/predict", "POST", "JSON API returning prediction, probabilities and model accuracy for all three targets"],
  ["/dashboard", "GET", "Dataset summary, model comparison table and confusion matrices"],
  ["/api/metrics", "GET", "Evaluation metrics in JSON"],
], [1900, 1300, 6438]));
c.push(cap("Table 4: Application routes"));
c.push(para("The app validates every input (allowed category values and date format) and returns HTTP 400 for invalid requests. Because the models are not reliable, the prediction page shows a visible notice and always displays each model's accuracy next to the baseline so that users do not mistake the output for real guidance."));

// ---------- discussion ----------
c.push(h1("7. Discussion and Limitations"));
c.push(bullet([b("No learnable signal. "), "The negative result comes from the data, not from the algorithms or the code. Tuning hyperparameters or trying deeper models would not help."]));
c.push(bullet([b("Limited features. "), "Only descriptive fields are available. Real predictors such as adulterant concentration, lab measurements, region, supplier history or inspector notes are missing."]));
c.push(bullet([b("Small sample and short period. "), "1,000 rows over six months cannot capture seasonality or rare events."]));
c.push(bullet([b("Label definitions. "), "It is unclear how severity and risk were assigned, and the dataset shows no link between them."]));
c.push(h1("8. Conclusion and Future Work"));
c.push(para("A complete, reproducible pipeline and web application were built, and the evaluation was done honestly against a baseline. The result is that this dataset cannot support predictive modelling: all models perform at chance level and statistical tests find no relationships. The system is best seen as a working template that can be retrained on better data by replacing the CSV."));
c.push(para("Recommended next steps:"));
["collect real records with measurable features (adulterant concentration, test readings, region, date of production);",
 "verify that product and category values are consistent;",
 "define severity and risk labels with clear rules so they relate to each other;",
 "re-run the pipeline, add feature importance analysis and consider cost-sensitive metrics, since missing a high-risk case is worse than a false alarm."].forEach(t => c.push(bullet(t)));
c.push(h1("Appendix: How to run"));
["pip install -r requirements.txt", "python train.py", "python app.py"].forEach(t =>
  c.push(new Paragraph({ spacing: { after: 40 }, shading: { type: ShadingType.CLEAR, fill: "F3F4F6", color: "auto" },
    children: [new TextRun({ text: t, font: "Courier New", size: 20 })] })));
c.push(para("Then open http://127.0.0.1:5000 in a browser. On Google Colab, the app is run in a background thread and displayed with output.serve_kernel_port_as_iframe(5000).", { spacing: { before: 120, after: 120, line: 300 } }));

const doc = new Document({
  creator: "", title: "Food Adulteration Analysis using Machine Learning",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Calibri", size: 30, bold: true, color: GREEN }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Calibri", size: 25, bold: true, color: "1F2933" }, paragraph: { outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({
      tabStops: [{ type: TabStopType.RIGHT, position: W }],
      children: [new TextRun({ text: "Food Adulteration Detection using Machine Learning", font: "Calibri", size: 16, color: "888888" }),
        new TextRun({ text: "\tPage ", font: "Calibri", size: 16, color: "888888" }),
        new TextRun({ children: [PageNumber.CURRENT], font: "Calibri", size: 16, color: "888888" })] })] }) },
    children: c,
  }],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync("/mnt/user-data/outputs/Food_Adulteration_ML_Report.docx", buf); console.log("ok"); });
