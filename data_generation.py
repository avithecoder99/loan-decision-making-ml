# create_synthetic_data.py
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

data = pd.DataFrame({
    "Name": [f"User_{i}" for i in range(n)],
    "Age": np.random.randint(21, 65, n),
    "Income": np.random.randint(25000, 150000, n),
    "Employment_Status": np.random.choice(["Employed", "Self-Employed", "Unemployed"], n),
    "Loan_Amount": np.random.randint(5000, 50000, n),
    "Credit_Score": np.random.randint(300, 850, n),
    "Debt_To_Income": np.round(np.random.uniform(0.1, 0.5, n), 2),
    "Loan_Term": np.random.choice([12, 24, 36, 60], n),
    "Address": [f"Address_{i}" for i in range(n)],
    "Contact_Info": [f"user{i}@email.com" for i in range(n)],
    "Eligibility": np.random.choice(["Eligible", "Not Eligible", "Needs Review"], n, p=[0.6, 0.3, 0.1])
})

data.to_csv("data/synthetic_loan_data.csv", index=False)
print(" Synthetic data saved to data/synthetic_loan_data.csv")
