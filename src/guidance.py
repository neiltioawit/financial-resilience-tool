def generate_guidance(
    predictions,
    expense_income_ratio,
    debt_payment_ratio,
    emergency_coverage
):
    """
    Generate rule-based educational financial guidance
    based on model predictions and key financial indicators.
    """

    guidance = []

    # Financial Health
    if predictions["financial_health"] == "Healthy":
        guidance.append(
            "Financial health appears healthy. Maintain current spending, "
            "saving, and financial management practices."
        )
    elif predictions["financial_health"] == "Moderate":
        guidance.append(
            "Financial health appears moderate. Review essential expenses "
            "and strengthen the balance between income, spending, and savings."
        )
    else:
        guidance.append(
            "Financial health appears vulnerable. Prioritize essential "
            "expense control, debt management, and building financial reserves."
        )

    # Debt Readiness
    if predictions["debt_readiness"] == "Ready":
        guidance.append(
            "Debt readiness appears favorable. Before taking on additional "
            "debt, assess whether the expected payment remains affordable "
            "under changes in income or expenses."
        )
    elif predictions["debt_readiness"] == "Caution":
        guidance.append(
            "Debt readiness suggests caution. Review existing debt payments "
            "and affordability before taking on additional borrowing."
        )
    else:
        guidance.append(
            "Debt readiness indicates high risk. Prioritize managing existing "
            "debt obligations and avoid unnecessary additional borrowing."
        )

    # Emergency Preparedness
    if predictions["emergency_preparedness"] == "Adequate":
        guidance.append(
            "Emergency preparedness appears adequate. Continue maintaining "
            "an accessible emergency reserve."
        )
    elif predictions["emergency_preparedness"] == "Developing":
        guidance.append(
            "Emergency preparedness is developing. Consider gradually "
            "increasing emergency savings to strengthen financial resilience."
        )
    else:
        guidance.append(
            "Emergency preparedness appears inadequate. Prioritize building "
            "an accessible emergency reserve for unexpected expenses or "
            "income disruptions."
        )

    # Ratio-based guidance
    if expense_income_ratio > 0.70:
        guidance.append(
            "Essential expenses consume a relatively large share of income. "
            "Review recurring expenses and identify opportunities to improve "
            "cash-flow flexibility."
        )

    if debt_payment_ratio > 0.30:
        guidance.append(
            "Debt payments represent a relatively large share of income. "
            "Review debt obligations and affordability carefully."
        )

    if emergency_coverage < 3:
        guidance.append(
            "Emergency savings cover less than three months of essential "
            "expenses based on the reported figures. Consider gradually "
            "building the reserve."
        )

    return guidance