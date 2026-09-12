import pandas as pd


DEPLOYMENT_FEATURES = [
    "age",
    "hh_size",
    "dependents",
    "income",
    "essential_exp",
    "debt_payment",
    "emergency_savings",
    "income_stability",
    "financial_literacy",
    "financial_management",
    "expense_income_ratio",
    "debt_payment_ratio",
    "emergency_coverage",
    "employment"
]


def prepare_model_input(respondent):
    """
    Convert raw respondent inputs into the exact
    14-feature structure expected by the EBM models.
    """

    income = respondent["income"]
    essential_exp = respondent["essential_exp"]
    debt_payment = respondent["debt_payment"]
    emergency_savings = respondent["emergency_savings"]

    # Calculate derived predictors
    expense_income_ratio = essential_exp / income
    debt_payment_ratio = debt_payment / income
    emergency_coverage = emergency_savings / essential_exp

    # Combine original and derived predictors
    model_input = {
        **respondent,
        "expense_income_ratio": expense_income_ratio,
        "debt_payment_ratio": debt_payment_ratio,
        "emergency_coverage": emergency_coverage
    }

    # Create DataFrame in the exact training feature order
    X = pd.DataFrame(
        [model_input],
        columns=DEPLOYMENT_FEATURES
    )

    return X