# Day 01 — Practical: Setup and First Look at Data

**Module:** Module 1: Foundations

## 1. Install and Check Python

> Note: These commands only work if Python is already installed. If it is not, install Python from python.org (or your OS package manager) first.

```bash
python --version
pip install --upgrade pip
```

Create a virtual environment (optional but recommended):

```bash
python -m venv ml-env
# Windows
ml-env\Scripts\activate
# macOS / Linux
source ml-env/bin/activate
```

Run the activation command for your operating system in the repository's
terminal. In Windows PowerShell, use `.\ml-env\Scripts\Activate.ps1`;
in Command Prompt, use `ml-env\Scripts\activate.bat`.

## 2. Install Core Libraries

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

If you installed the course requirements with `pip install -r requirements.txt`,
you already have these packages. Install them in the activated environment so
Jupyter uses the same Python that has the libraries.

Check that they import correctly:

```python
import numpy, pandas, matplotlib, sklearn
print("All libraries loaded")
```

## 3. What Each Library Does

- **NumPy:** fast numbers, arrays, math.
- **pandas:** tables (DataFrames), reading CSV files, data exploration.
- **Matplotlib:** plots and charts.
- **scikit-learn:** ready-made ML models and tools.
- **Jupyter:** interactive notebooks (or use Google Colab in the browser).

## 4. Start and Use Jupyter Notebook

1. Open a terminal and change directory to the repository folder (the folder
   containing `requirements.txt`). Activate `ml-env` if you created it.
2. Start Jupyter:

   ```bash
   jupyter notebook
   ```

   If that command is not found, try `python -m notebook`. Jupyter normally
   opens a browser tab showing the files in the current folder. If it does not,
   copy the local URL printed in the terminal into your browser. Keep the
   terminal running while you use the notebook.
3. In the Jupyter file page, navigate to
   `00-foundations/day-01-introduction-to-ml/`, then choose **New > Python 3**
   (the wording can vary slightly by Jupyter version). Rename the notebook,
   for example `day01-practice.ipynb`.
4. A notebook is made of cells. Select a cell, choose **Code** or **Markdown**
   from the cell-type menu, and run it with **Shift+Enter**. Code cells run
   Python; Markdown cells hold notes. Run code from top to bottom, because later
   cells can use variables created earlier.
5. Copy each Python example below into its own code cell and run it. The output
   appears underneath the cell. To rerun after changing code, select the cell
   and press **Shift+Enter** again. Use **File > Save Notebook** (or Ctrl+S) to
   save your work.
6. When finished, save the notebook, then stop Jupyter from the terminal with
   **Ctrl+C** and confirm if prompted.

Alternatively, open [Google Colab](../../../docs/setup/google-colab.md) in a
browser and create a new notebook. Paste and run the same code cells there; no
local Jupyter installation is needed.

## 5. Load a Small Dataset and Explore

```python
import pandas as pd

data = {
    "Age": [22, 25, 35, 40, 30],
    "Income": [30000, 40000, 80000, 90000, 70000],
    "Previous_Purchase": ["No", "No", "Yes", "Yes", "Yes"],
    "Bought_Laptop": ["No", "No", "Yes", "Yes", "Yes"],
}
df = pd.DataFrame(data)

print(df.head())
print(df.shape)          # rows, columns
print(df.dtypes)         # data types
print(df.isnull().sum()) # missing values
print(df.describe())     # quick stats
```

## 6. Rule-Based Program vs Learned Model

Rule-based (human writes the rule):

```python
def rule_based(row):
    if row["Income"] >= 70000 and row["Previous_Purchase"] == "Yes":
        return "Buy"
    return "No"

print(df.apply(rule_based, axis=1))
```

Learned model (scikit-learn learns the rule from data):

```python
from sklearn.tree import DecisionTreeClassifier

X = df[["Age", "Income"]]
y = (df["Bought_Laptop"] == "Yes").astype(int)

model = DecisionTreeClassifier(random_state=0)
model.fit(X, y)

new_customer = pd.DataFrame({"Age": [32], "Income": [75000]})
prediction = model.predict(new_customer)[0]
print("Buy" if prediction == 1 else "No")
```

Compare the two: the rule-based version applies conditions written by a person;
the learned model fits a decision tree to the examples. This dataset has only
five rows and two input features, so it is a tiny demonstration, not a reliable
way to make real purchasing decisions. The rule and model also use different
inputs: the rule checks previous purchase history, while the model uses age and
income.

## 7. Real-World Application

- Spam filtering: subject/words → spam or not.
- Loan approval: income/credit history → approve or not.
- Recommendation systems: watch history → what to suggest next.

## Outcome

Learners can set up Python ML tools, explore a small dataset, and run both a simple rule and a tiny learned model.
