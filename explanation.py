import pandas as pd


def get_local_ebm_explanation(model, X):
    """
    Generate local EBM contributions for the predicted class.
    """

    contributions = model.eval_terms(X)[0]

    predicted_class = model.predict(X)[0]
    predicted_class_index = list(model.classes_).index(predicted_class)

    explanation_df = pd.DataFrame({
        "term": model.term_names_,
        "contribution": contributions[:, predicted_class_index]
    })

    explanation_df["abs_contribution"] = (
        explanation_df["contribution"].abs()
    )

    explanation_df = explanation_df.sort_values(
        "abs_contribution",
        ascending=False
    ).reset_index(drop=True)

    return {
        "prediction": str(predicted_class),
        "explanation": explanation_df
    }


def generate_all_explanations(models, X):
    """
    Generate local EBM explanations for all three targets.
    """

    explanations = {}

    for target, model in models.items():
        explanations[target] = get_local_ebm_explanation(
            model,
            X
        )

    return explanations