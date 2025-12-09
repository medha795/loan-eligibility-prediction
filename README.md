## 📂 Data Sources

The raw LendingClub dataset is large and is not stored in the repository.

- **LendingClub full loan data (accepted & rejected applications)**  
  [Kaggle – Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

- **LendingClub data dictionary and schema**  
  [OpenIntro – loans_full_schema](https://www.openintro.org/data/index.php?data=loans_full_schema)

- **Combined processed dataset for modeling**  
  [Google Drive – combined_loan_data_processed.csv](https://drive.google.com/file/d/1MgP0eP6GZAfoTcxsLtSKJpqj3pl5xKAz/view?usp=drive_link)

---

### 📁 Raw Data Location

Place the following files into `data/raw/`:

- `accepted_2007_to_2018Q4.csv`
- `rejected_2007_to_2018Q4.csv`

These files are ignored from Git tracking to keep the repository lightweight.

diff --git a/reports/dataset_comparison.md b/reports/dataset_comparison.md
new file mode 100644
index 0000000000000000000000000000000000000000..ea6088d22bdc933e8fdaa185043b4783b6e43b63
--- /dev/null
+++ b/reports/dataset_comparison.md
@@ -0,0 +1,40 @@
+# Engineered vs. Combined Processed Loan Datasets
+
+This document summarizes how the **engineered_loan_dataset** improves upon the **combined_loan_dataset_processed** file for modeling loan eligibility. The engineered version applies transformations implemented in `prepare_data.py` to make the data more suitable for machine-learning workflows.
+
+## Source and Target
+- **Source:** `combined_loan_dataset_processed.csv` is the merged LendingClub file that retains raw string encodings for terms, employment length, percentages, and category labels. It may include inconsistent casing and missing values.
+- **Engineered output:** `engineered_loan_dataset.csv` is the model-ready table produced by the chunked pipeline. It contains standardized numeric fields, one-hot encoded categoricals, and a binary target label.
+
+## Key Engineering Steps
+
+### Column normalization
+- Lowercases and trims all column headers for consistent downstream handling before any other processing.【F:prepare_data.py†L99-L125】
+
+### Numeric feature construction
+- Converts text fields to numeric equivalents:
+  - `term` → `term_months` by extracting the month count from strings like "36 months".【F:prepare_data.py†L127-L140】
+  - `emp_length` → `emp_length_years` by normalizing strings (e.g., "< 1 year" → 0, "10+ years" → 10).【F:prepare_data.py†L127-L140】
+  - Percentage strings in `int_rate` and `dti` are stripped of `%` and cast to floats.【F:prepare_data.py†L134-L139】
+- Builds a single `fico_score` by averaging the high and low bounds when both are present (or using whichever is available).【F:prepare_data.py†L84-L95】【F:prepare_data.py†L134-L141】
+
+### Target label mapping
+- Maps the text `loan_status` label to a binary indicator (`accepted` → 1, `rejected` → 0) and drops rows with invalid labels, ensuring the engineered set only contains usable targets.【F:prepare_data.py†L142-L155】
+
+### Missing-value handling
+- For numeric columns, fills `NaN` values with the column median to keep distributions stable without creating extreme values.【F:prepare_data.py†L155-L173】
+- For categorical columns, trims whitespace, converts placeholder strings "nan" to nulls, and imputes with the mode (or "unknown" if no mode exists).【F:prepare_data.py†L162-L180】
+
+### Categorical encoding
+- One-hot encodes all remaining categorical features without dropping the first level, preserving full information for models that do not require dummy-variable dropping.【F:prepare_data.py†L181-L199】
+
+### Schema alignment and scalability
+- Processes the massive source file in chunks: a large initial sample establishes the full schema, then each subsequent chunk is engineered and reindexed to that schema before being appended. This prevents missing columns and keeps memory usage manageable on 30M+ rows.【F:prepare_data.py†L203-L299】
+
+## Net Improvements for Modeling
+- **Consistent numeric inputs:** Percentage and tenure fields become numeric, enabling algorithms to learn gradients from them instead of parsing strings.
+- **Reduced leakage and noise:** Invalid target labels are removed early, and all features are standardized to lower-case, reducing mismatches from inconsistent casing.
+- **Robust handling of sparsity:** Median/mode imputation keeps rows usable while minimizing distribution shifts; categorical dummies ensure models receive explicit indicators for every level observed in the sample.
+- **Operational readiness:** Chunked processing yields a single, schema-stable CSV that fits in memory when consumed by training pipelines, while still originating from the large combined raw dataset.
+
+These steps collectively transform the raw combined data into an optimized feature matrix tailored for training loan eligibility prediction models.
