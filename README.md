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


## Engineered vs. Combined Processed Loan Datasets

This section summarizes how the engineered_loan_dataset.csv improves upon the combined_loan_dataset_processed.csv file for modeling loan eligibility.
The engineered version applies the transformations implemented in prepare_dataset.py to generate a dataset that is clean, consistent, and optimized for machine-learning workflows.

### 📌 Source and Target

Source:
combined_loan_dataset_processed.csv

The merged LendingClub dataset containing raw string encodings for:

- loan terms

- employment length

- percentage fields

- categorical labels

It may include inconsistent casing, missing values, and non-standardized formats.

### Engineered Output:
engineered_loan_dataset.csv

A model-ready dataset produced by the chunked feature-engineering pipeline.
It contains:

- standardized numeric fields

- one-hot encoded categorical indicators

- a binary target label

### ⚙️ Key Engineering Steps
#### 1. Column Normalization

Lowercases and trims all column headers to ensure consistent downstream handling.
(Implemented in prepare_dataset.py)

#### 2. Numeric Feature Construction

Transforms textual fields into usable numeric inputs:

term → term_months
Extracts the month count from strings like "36 months".

emp_length → emp_length_years
Normalizes employment length strings:

- "< 1 year" → 0

- "10+ years" → 10

- "3 years" → 3

Percentage fields (int_rate, dti)
Strips % signs and converts to float.

fico_score construction
Creates a single score by averaging:

(fico_range_low + fico_range_high) / 2


or using whichever value is available.

#### 3. Target Label Mapping

Maps textual loan_status values to numeric:

"accepted" → 1

"rejected" → 0

Drops rows with invalid or unmappable values, ensuring a clean supervised learning setup.

#### 4. Missing-Value Handling

Numeric Features:
Impute with the median, preserving central tendencies without extreme values.

Categorical Features:

Trim whitespace

Convert "nan" strings to true nulls

Impute missing values with the mode (or "unknown" as fallback)

#### 5. Categorical Encoding

One-hot encodes all categorical features.

No levels are dropped (drop_first=False), preserving full information for downstream ML models such as tree-based algorithms.

#### 6. Schema Alignment & Scalability (Chunked Processing)

Because the original dataset exceeds 30 million rows, the pipeline:

Reads a large sample (e.g., 500k rows)

Engineers features → establishes the full schema

Reads the rest of the dataset in chunks

Processes each chunk and aligns columns to the established schema

Appends rows incrementally to the engineered output file

This approach ensures:

consistent column ordering

guaranteed presence of all expected dummy variables

minimal RAM usage

### 🚀 Net Improvements for Modeling
✔ Consistent Numeric Inputs

Previously string-encoded fields (percentages, durations, employment length) become proper numerical features enabling gradients and comparisons.

✔ Cleaner & More Reliable Target Labels

Invalid or ambiguous loan_status entries are removed, improving downstream training quality.

✔ Robust Missing-Value Strategy

Median/mode imputation prevents sample loss while preserving feature distributions.

✔ High-Quality Categorical Representation

All categorical levels are explicitly represented via one-hot encoding — essential for interpretability and model stability.

✔ Operationally Scalable

Chunked processing allows the dataset to originate from a massive 30M+ row file while producing a final engineered dataset that:

fits into memory

is ML-ready

maintains schema consistency
