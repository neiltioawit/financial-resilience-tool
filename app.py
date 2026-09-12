import streamlit as st

from src.preprocessing import prepare_model_input
from src.prediction import load_models, generate_predictions
from src.explanation import generate_all_explanations
from src.guidance import generate_guidance


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Financial Resilience Decision-Support Tool",
    page_icon="💰",
    layout="wide"
)


# ==================================================
# CUSTOM STYLING
# ==================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.08rem;
        color: #555555;
        margin-bottom: 1.2rem;
        line-height: 1.6;
    }

    .section-title {
        font-size: 1.65rem;
        font-weight: 700;
        margin-top: 1.6rem;
        margin-bottom: 0.7rem;
    }

    .subsection-title {
        font-size: 1.15rem;
        font-weight: 650;
        margin-top: 0.8rem;
        margin-bottom: 0.5rem;
    }

    .info-box {
        padding: 1.2rem 1.4rem;
        border-radius: 10px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
        margin-bottom: 1.2rem;
        line-height: 1.6;
    }

    .privacy-box {
        padding: 1.2rem 1.4rem;
        border-radius: 10px;
        border: 1px solid #d8d8d8;
        background-color: #f8f8f8;
        margin-bottom: 1.4rem;
        line-height: 1.6;
    }

    .result-box {
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
        min-height: 160px;
    }

    .result-title {
        font-size: 1.05rem;
        font-weight: 650;
        margin-bottom: 0.7rem;
    }

    .result-value {
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .result-description {
        font-size: 0.9rem;
        color: #666666;
        line-height: 1.5;
    }

    .guidance-item {
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
        border-radius: 9px;
        border: 1px solid #e0e0e0;
        background-color: #fafafa;
        line-height: 1.55;
    }

    .small-note {
        font-size: 0.88rem;
        color: #666666;
        line-height: 1.5;
    }

    div.stButton > button {
        width: 100%;
        min-height: 3.5rem;
        font-size: 1.15rem;
        font-weight: 650;
        border-radius: 9px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# LOAD TRAINED MODELS
# ==================================================

models = load_models()


# ==================================================
# USER-FACING TERM NAMES
# ==================================================

TERM_LABELS = {
    "age": "Age",
    "hh_size": "Household Size",
    "dependents": "Number of Dependents",
    "income": "Monthly Household Income",
    "essential_exp": "Monthly Essential Expenses",
    "debt_payment": "Monthly Debt Payments",
    "emergency_savings": "Emergency Savings",
    "income_stability": "Income Stability",
    "financial_literacy": "Financial Literacy",
    "financial_management": "Financial Management",
    "expense_income_ratio": "Expense-to-Income Ratio",
    "debt_payment_ratio": "Debt Payment Ratio",
    "emergency_coverage": "Emergency Coverage",
    "employment": "Employment Status"
}


TERM_DEFINITIONS = {
    "Age":
        "Age of the respondent in years.",

    "Household Size":
        "Total number of people living in the household.",

    "Number of Dependents":
        "Number of household members who depend financially on household income.",

    "Monthly Household Income":
        "Total household income received per month.",

    "Monthly Essential Expenses":
        "Monthly spending on necessary household needs such as food, housing, utilities, and other essential expenses.",

    "Monthly Debt Payments":
        "Total amount paid each month toward loans, credit, and other debt obligations.",

    "Emergency Savings":
        "Funds currently available to help cover unexpected expenses or temporary income disruptions.",

    "Income Stability":
        "Self-rated stability of household income, from 1 (very unstable) to 5 (very stable).",

    "Financial Literacy":
        "Self-rated ability to understand and apply basic financial concepts, from 1 (very low) to 5 (very high).",

    "Financial Management":
        "Self-rated ability to manage spending, saving, budgeting, and financial obligations, from 1 (very poor) to 5 (very good).",

    "Expense-to-Income Ratio":
        "Essential household expenses divided by monthly household income. A higher value means a larger share of income is allocated to essential expenses.",

    "Debt Payment Ratio":
        "Monthly debt payments divided by monthly household income. A higher value indicates that a larger share of income is committed to debt payments.",

    "Emergency Coverage":
        "Emergency savings divided by monthly essential expenses. The result approximates the number of months of essential expenses that current emergency savings could cover.",

    "Employment Status":
        "Current employment category reported by the respondent."
}


def format_term_name(term):

    if term in TERM_LABELS:
        return TERM_LABELS[term]

    if " & " in term:

        parts = term.split(" & ")

        return " × ".join(
            TERM_LABELS.get(
                part,
                part.replace("_", " ").title()
            )
            for part in parts
        )

    return term.replace("_", " ").title()


def get_term_definition(term):

    formatted_term = format_term_name(term)

    if formatted_term in TERM_DEFINITIONS:
        return TERM_DEFINITIONS[formatted_term]

    if " × " in formatted_term:
        parts = formatted_term.split(" × ")

        definitions = [
            TERM_DEFINITIONS.get(
                part,
                f"Model interaction involving {part}."
            )
            for part in parts
        ]

        return (
            "An interaction term representing how the model combines "
            "the effects of " + " and ".join(parts) + ". "
            + " ".join(definitions)
        )

    return "Model feature used in generating the assessment."


# ==================================================
# OUTCOME DEFINITIONS
# ==================================================

OUTCOME_DEFINITIONS = {

    "Financial Health": {
        "Healthy":
            "The household's reported financial characteristics are most consistent with a relatively strong overall financial position.",

        "Moderate":
            "The household's reported financial characteristics indicate a middle-range financial position, with some areas that may benefit from monitoring or improvement.",

        "Vulnerable":
            "The household's reported financial characteristics indicate greater financial vulnerability, suggesting that cash-flow pressure, debt obligations, or limited reserves may require attention."
    },

    "Debt Readiness": {
        "Ready":
            "The household's reported financial characteristics are most consistent with a comparatively favorable position for managing debt-related obligations.",

        "Caution":
            "The household's reported financial characteristics suggest that additional borrowing should be considered carefully, particularly in relation to existing obligations and affordability.",

        "High Risk":
            "The household's reported financial characteristics indicate greater difficulty or potential strain in managing debt-related obligations."
    },

    "Emergency Preparedness": {
        "Adequate":
            "The household appears comparatively better positioned to absorb unexpected expenses or temporary income disruptions.",

        "Developing":
            "The household has some capacity to handle unexpected financial events, but additional emergency reserves may strengthen resilience.",

        "Inadequate":
            "The household appears to have limited financial capacity to absorb unexpected expenses or income disruptions."
    }
}


# ==================================================
# APPLICATION HEADER
# ==================================================

st.markdown(
    '<div class="main-title">Financial Resilience Decision-Support Tool</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        An explainable machine learning tool for assessing household
        financial health, debt readiness, and emergency preparedness.
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# ABOUT THE ASSESSMENT
# ==================================================

st.markdown(
    """
    <div class="info-box">

    <strong>About the Assessment</strong>

    <p>
    This tool uses explainable machine learning to provide estimates
    across three dimensions of household financial resilience.
    The assessment considers household characteristics, financial
    conditions, financial behaviors, and self-reported financial
    characteristics provided by the user.
    </p>

    <strong>Financial Health</strong><br>
    An estimate of the household's overall financial condition based
    on the characteristics provided.

    <br><br>

    <strong>Debt Readiness</strong><br>
    An estimate related to the household's financial position and
    ability to manage debt-related obligations.

    <br><br>

    <strong>Emergency Preparedness</strong><br>
    An estimate of the household's capacity to withstand unexpected
    financial expenses or income disruptions.

    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# PRIVACY & INFORMATION
# ==================================================

st.markdown(
    '<div class="subsection-title">Privacy & Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="privacy-box">

    <strong>No personally identifying information is requested.</strong>

    <p>
    This assessment does not ask for your name, home address,
    phone number, email address, bank account information,
    identification numbers, or other direct personal identifiers.
    </p>

    <p>
    The requested information is limited to household, financial,
    employment, and self-assessed financial characteristics that
    are used as inputs to the assessment models.
    </p>

    <p class="small-note">
    Please provide information only to the extent that you are
    comfortable doing so.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# HOUSEHOLD FINANCIAL INFORMATION
# ==================================================

st.markdown(
    '<div class="section-title">Household Financial Information</div>',
    unsafe_allow_html=True
)

st.write(
    "Provide the information below to generate your financial "
    "resilience assessment. Hover over the information icons "
    "beside individual fields for additional definitions."
)


# ==================================================
# TWO-COLUMN QUESTIONNAIRE
# ==================================================

left_col, right_col = st.columns(2)


with left_col:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1,
        help="Age of the respondent in years."
    )

    hh_size = st.number_input(
        "Household Size",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
        help="Total number of people living in the household."
    )

    dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=20,
        value=2,
        step=1,
        help="Number of household members who depend financially on household income."
    )

    income = st.number_input(
        "Monthly Household Income (₱)",
        min_value=1.0,
        value=50000.0,
        step=1000.0,
        help="Total household income received per month."
    )

    essential_exp = st.number_input(
        "Monthly Essential Expenses (₱)",
        min_value=1.0,
        value=30000.0,
        step=1000.0,
        help="Monthly spending on necessary household needs such as food, housing, utilities, and other essential expenses."
    )

    debt_payment = st.number_input(
        "Monthly Debt Payments (₱)",
        min_value=0.0,
        value=5000.0,
        step=500.0,
        help="Total amount paid each month toward loans, credit, and other debt obligations."
    )


with right_col:

    emergency_savings = st.number_input(
        "Emergency Savings (₱)",
        min_value=0.0,
        value=100000.0,
        step=1000.0,
        help="Funds currently available to help cover unexpected expenses or temporary income disruptions."
    )

    income_stability = st.slider(
        "Income Stability",
        min_value=1,
        max_value=5,
        value=4,
        help="Self-rated stability of household income. 1 = Very unstable; 5 = Very stable."
    )

    financial_literacy = st.slider(
        "Financial Literacy",
        min_value=1,
        max_value=5,
        value=4,
        help="Self-rated ability to understand and apply basic financial concepts. 1 = Very low; 5 = Very high."
    )

    financial_management = st.slider(
        "Financial Management",
        min_value=1,
        max_value=5,
        value=4,
        help="Self-rated ability to manage spending, saving, budgeting, and financial obligations. 1 = Very poor; 5 = Very good."
    )

    employment = st.selectbox(
        "Employment Status",
        options=[
            "Employed",
            "Self-employed",
            "Unemployed"
        ],
        help="Current employment category reported by the respondent."
    )


# ==================================================
# ASSESSMENT BUTTON
# ==================================================

st.markdown("<br>", unsafe_allow_html=True)

assess_clicked = st.button(
    "Assess Financial Resilience",
    type="primary"
)


# ==================================================
# RUN ASSESSMENT
# ==================================================

if assess_clicked:

    respondent = {
        "age": age,
        "hh_size": hh_size,
        "dependents": dependents,
        "income": income,
        "essential_exp": essential_exp,
        "debt_payment": debt_payment,
        "emergency_savings": emergency_savings,
        "income_stability": income_stability,
        "financial_literacy": financial_literacy,
        "financial_management": financial_management,
        "employment": employment
    }


    # ==================================================
    # PREPARE MODEL INPUT
    # ==================================================

    X = prepare_model_input(respondent)


    # ==================================================
    # PREDICTIONS
    # ==================================================

    prediction_results = generate_predictions(
        models,
        X
    )

    predictions = {
        target: result["prediction"]
        for target, result in prediction_results.items()
    }


    # ==================================================
    # LOCAL EXPLANATIONS
    # ==================================================

    explanations = generate_all_explanations(
        models,
        X
    )


    # ==================================================
    # DERIVED INDICATORS
    # ==================================================

    expense_income_ratio = float(
        X["expense_income_ratio"].iloc[0]
    )

    debt_payment_ratio = float(
        X["debt_payment_ratio"].iloc[0]
    )

    emergency_coverage = float(
        X["emergency_coverage"].iloc[0]
    )


    # ==================================================
    # GUIDANCE
    # ==================================================

    guidance = generate_guidance(
        predictions,
        expense_income_ratio,
        debt_payment_ratio,
        emergency_coverage
    )


    # ==================================================
    # RESULTS
    # ==================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Your Financial Resilience Results</div>',
        unsafe_allow_html=True
    )

    st.write(
        "These results are model-based estimates derived from "
        "the information you provided."
    )


    # ==================================================
    # RESULT SUMMARY
    # ==================================================

    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.markdown(
            f"""
            <div class="result-box">

            <div class="result-title">
                Financial Health
            </div>

            <div class="result-value">
                {predictions["financial_health"]}
            </div>

            <div class="result-description">
                Estimated overall household financial condition.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("ⓘ What does this result mean?"):

            definition = OUTCOME_DEFINITIONS[
                "Financial Health"
            ][predictions["financial_health"]]

            st.write(definition)


    with result_col2:

        st.markdown(
            f"""
            <div class="result-box">

            <div class="result-title">
                Debt Readiness
            </div>

            <div class="result-value">
                {predictions["debt_readiness"]}
            </div>

            <div class="result-description">
                Estimated ability to manage debt-related obligations.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("ⓘ What does this result mean?"):

            definition = OUTCOME_DEFINITIONS[
                "Debt Readiness"
            ][predictions["debt_readiness"]]

            st.write(definition)


    with result_col3:

        st.markdown(
            f"""
            <div class="result-box">

            <div class="result-title">
                Emergency Preparedness
            </div>

            <div class="result-value">
                {predictions["emergency_preparedness"]}
            </div>

            <div class="result-description">
                Estimated capacity to withstand unexpected
                financial disruptions.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("ⓘ What does this result mean?"):

            definition = OUTCOME_DEFINITIONS[
                "Emergency Preparedness"
            ][predictions["emergency_preparedness"]]

            st.write(definition)


    # ==================================================
    # MODEL PROBABILITIES
    # ==================================================

    st.markdown(
        '<div class="section-title">Model Probabilities</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The probabilities indicate how strongly the model favors "
        "each possible outcome category for the information provided. "
        "They should be interpreted as model estimates rather than "
        "guarantees of future financial outcomes."
    )


    # --------------------------------------------------
    # THREE COLUMNS — ONE ROW
    # --------------------------------------------------

    prob_col1, prob_col2, prob_col3 = st.columns(3)


    with prob_col1:

        st.subheader("Financial Health")

        for label, probability in prediction_results[
            "financial_health"
        ]["probabilities"].items():

            st.progress(
                float(probability),
                text=f"{label}: {probability:.1%}"
            )

            with st.expander(f"ⓘ {label}"):

                st.write(
                    OUTCOME_DEFINITIONS[
                        "Financial Health"
                    ][label]
                )


    with prob_col2:

        st.subheader("Debt Readiness")

        for label, probability in prediction_results[
            "debt_readiness"
        ]["probabilities"].items():

            st.progress(
                float(probability),
                text=f"{label}: {probability:.1%}"
            )

            with st.expander(f"ⓘ {label}"):

                st.write(
                    OUTCOME_DEFINITIONS[
                        "Debt Readiness"
                    ][label]
                )


    with prob_col3:

        st.subheader("Emergency Preparedness")

        for label, probability in prediction_results[
            "emergency_preparedness"
        ]["probabilities"].items():

            st.progress(
                float(probability),
                text=f"{label}: {probability:.1%}"
            )

            with st.expander(f"ⓘ {label}"):

                st.write(
                    OUTCOME_DEFINITIONS[
                        "Emergency Preparedness"
                    ][label]
                )


    # ==================================================
    # WHY DID THE MODEL MAKE THESE PREDICTIONS?
    # ==================================================

    st.markdown(
        '<div class="section-title">Why Did the Model Make These Predictions?</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The Explainable Boosting Machine (EBM) provides a local "
        "explanation of each prediction by showing which model terms "
        "contributed most strongly to the predicted category."
    )

    st.info(
        "How to read the Model Influence Score: positive values "
        "support the predicted outcome, while negative values work "
        "against it. The score is an additive model-scale value. "
        "It is not a percentage and should not be interpreted as "
        "a percentage change in probability."
    )


    # ==================================================
    # EXPLANATION HELPER
    # ==================================================

    def display_explanation(target):

        explanation_df = explanations[
            target
        ]["explanation"].copy()

        top_factors = explanation_df.head(5)[
            ["term", "contribution"]
        ].copy()

        top_factors["Factor"] = top_factors[
            "term"
        ].apply(format_term_name)

        top_factors["Definition"] = top_factors[
            "term"
        ].apply(get_term_definition)

        top_factors["Direction"] = top_factors[
            "contribution"
        ].apply(
            lambda x:
                "Supports prediction"
                if x > 0
                else "Works against prediction"
        )

        top_factors["Model Influence Score"] = (
            top_factors["contribution"].round(3)
        )

        display_df = top_factors[
            [
                "Factor",
                "Direction",
                "Model Influence Score"
            ]
        ]

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "Hover over the information icon below to review "
            "how the model terms are defined."
        )

        for _, row in top_factors.iterrows():

            factor = row["Factor"]
            definition = row["Definition"]

            score = row["Model Influence Score"]

            direction = row["Direction"]

            st.markdown(
                f"**ⓘ {factor}** — {direction} "
                f"({score:+.3f})"
            )

            st.caption(definition)


    # ==================================================
    # TWO-COLUMN EXPLANATIONS
    # ==================================================

    explanation_col1, explanation_col2 = st.columns(2)


    with explanation_col1:

        st.subheader("Financial Health")

        display_explanation(
            "financial_health"
        )


    with explanation_col2:

        st.subheader("Debt Readiness")

        display_explanation(
            "debt_readiness"
        )


    st.subheader("Emergency Preparedness")

    display_explanation(
        "emergency_preparedness"
    )


    # ==================================================
    # SUGGESTED FINANCIAL ACTIONS
    # ==================================================

    st.markdown(
        '<div class="section-title">Suggested Financial Actions</div>',
        unsafe_allow_html=True
    )

    st.write(
        "These suggestions translate the assessment into practical "
        "areas for consideration. They are educational recommendations "
        "based on the model results and selected financial indicators; "
        "they are not personalized financial advice."
    )


    # --------------------------------------------------
    # ACTION INTRODUCTION
    # --------------------------------------------------

    st.info(
        "The suggested actions are intended to help users identify "
        "areas that may deserve attention. They should be considered "
        "alongside the household's actual financial circumstances."
    )


    # --------------------------------------------------
    # TWO-COLUMN GUIDANCE
    # --------------------------------------------------

    guidance_col1, guidance_col2 = st.columns(2)


    for index, item in enumerate(guidance):

        target_col = (
            guidance_col1
            if index % 2 == 0
            else guidance_col2
        )

        with target_col:

            st.markdown(
                f"""
                <div class="guidance-item">
                    <strong>Action {index + 1}</strong><br><br>
                    {item}
                </div>
                """,
                unsafe_allow_html=True
            )


    # ==================================================
    # FINANCIAL INDICATORS
    # ==================================================

    st.markdown(
        '<div class="subsection-title">Key Financial Indicators</div>',
        unsafe_allow_html=True
    )

    indicator_col1, indicator_col2, indicator_col3 = st.columns(3)


    with indicator_col1:

        st.metric(
            "Expense-to-Income Ratio",
            f"{expense_income_ratio:.1%}"
        )

        st.caption(
            "Essential expenses relative to household income."
        )


    with indicator_col2:

        st.metric(
            "Debt Payment Ratio",
            f"{debt_payment_ratio:.1%}"
        )

        st.caption(
            "Monthly debt payments relative to household income."
        )


    with indicator_col3:

        st.metric(
            "Emergency Coverage",
            f"{emergency_coverage:.1f} months"
        )

        st.caption(
            "Approximate months of essential expenses covered "
            "by reported emergency savings."
        )


    # ==================================================
    # HOW TO USE THE RESULTS
    # ==================================================

    st.markdown(
        '<div class="section-title">How to Use the Results</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The assessment is designed to be used as a structured "
        "reflection and decision-support tool rather than as a "
        "stand-alone financial decision."
    )


    use_col1, use_col2 = st.columns(2)


    with use_col1:

        st.markdown(
            """
            <div class="info-box">

            <strong>1. Review the predicted outcomes</strong>

            <p>
            Start with the three assessment areas: Financial Health,
            Debt Readiness, and Emergency Preparedness. These provide
            a high-level view of how the model classifies the household
            based on the information provided.
            </p>

            <strong>2. Examine the model probabilities</strong>

            <p>
            Review the probability distribution for each outcome.
            A high probability for one category indicates that the
            model strongly favors that category relative to the
            alternatives. A more distributed set of probabilities
            indicates greater model uncertainty between categories.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with use_col2:

        st.markdown(
            """
            <div class="info-box">

            <strong>3. Review the model explanations</strong>

            <p>
            Examine the factors with the strongest model influence.
            Positive contributions support the predicted category,
            while negative contributions work against it. These
            explanations describe how the model arrived at its
            prediction; they do not establish cause-and-effect
            relationships.
            </p>

            <strong>4. Consider the suggested actions</strong>

            <p>
            Use the suggested actions to identify areas that may
            warrant further review. Consider the recommendations
            together with actual household circumstances, financial
            goals, obligations, and constraints.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ==================================================
    # IMPORTANT NOTE
    # ==================================================

    st.warning(
        "Important Note: Model predictions are estimates, not "
        "financial advice. Actual financial circumstances may "
        "differ from the model's assessment."
    )


    # ==================================================
    # MODEL LIMITATION
    # ==================================================

    with st.expander("ⓘ About the Model and Its Limitations"):

        st.write(
            "The assessment is based on machine learning models "
            "developed from research data. The models identify "
            "patterns in the characteristics represented in the "
            "training data and use those patterns to generate "
            "estimates for new inputs."
        )

        st.write(
            "A prediction should therefore not be interpreted as "
            "a diagnosis, guarantee, or definitive judgment about "
            "a household's financial situation."
        )

        st.write(
            "Model explanations describe the contribution of "
            "model terms to a particular prediction. They should "
            "not be interpreted as evidence that changing a single "
            "factor will necessarily cause the predicted outcome "
            "to change."
        )