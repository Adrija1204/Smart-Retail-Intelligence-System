"""Streamlit dashboard for Smart Retail Intelligence reports."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT_DIRECTORY = Path(__file__).resolve().parents[1]


def read_report(relative_path: str) -> pd.DataFrame | None:
    """Read a generated report or show a dashboard warning when it is absent."""
    report_path = ROOT_DIRECTORY / relative_path
    if not report_path.exists():
        st.warning(f"Missing report: {relative_path}. Run the relevant pipeline first.")
        return None
    return pd.read_csv(report_path)


def show_eda() -> None:
    """Render EDA KPI and monthly-revenue sections."""
    kpis = read_report("reports/eda/kpis.csv")
    monthly = read_report("reports/eda/monthly_revenue.csv")
    if kpis is not None:
        values = kpis.iloc[0]
        columns = st.columns(4)
        columns[0].metric("Revenue", f"{values['revenue']:,.0f}")
        columns[1].metric("Orders", f"{values['order_count']:,.0f}")
        columns[2].metric("Customers", f"{values['customer_count']:,.0f}")
        columns[3].metric(
            "Average Order Value", f"{values['average_order_value']:,.2f}"
        )
    if monthly is not None:
        st.plotly_chart(
            px.line(monthly, x="month", y="revenue", title="Monthly Revenue"),
            use_container_width=True,
        )


def show_segments() -> None:
    """Render segment profiles and their relative customer value."""
    profiles = read_report("reports/segmentation/segment_profiles.csv")
    if profiles is None:
        return
    st.plotly_chart(
        px.bar(
            profiles,
            x="segment",
            y="average_monetary",
            color="customer_count",
            title="Average Customer Spend by Segment",
        ),
        use_container_width=True,
    )
    st.dataframe(profiles, use_container_width=True, hide_index=True)


def show_predictions() -> None:
    """Render model evaluation, top predicted buyers, and SHAP importance."""
    evaluation = read_report("reports/modeling/model_evaluation.csv")
    propensity = read_report("reports/modeling/customer_propensity.csv")
    importance = read_report("reports/modeling/shap_global_importance.csv")
    if evaluation is not None:
        st.subheader("Future-Purchase Model Evaluation")
        st.dataframe(evaluation, use_container_width=True, hide_index=True)
    if propensity is not None:
        st.subheader("Top Customers by Purchase Probability")
        st.dataframe(propensity.head(20), use_container_width=True, hide_index=True)
    if importance is not None:
        st.subheader("SHAP Global Feature Importance")
        st.plotly_chart(
            px.bar(
                importance.sort_values("mean_absolute_shap"),
                x="mean_absolute_shap",
                y="feature",
                orientation="h",
            ),
            use_container_width=True,
        )


def main() -> None:
    """Configure and render the Smart Retail Intelligence dashboard."""
    st.set_page_config(page_title="Smart Retail Intelligence", layout="wide")
    st.title("Smart Retail Intelligence System")
    st.caption("EDA, customer segmentation, and time-aware purchase propensity")
    show_eda()
    st.divider()
    st.header("Customer Segmentation")
    show_segments()
    st.divider()
    st.header("Purchase Propensity and Explainability")
    show_predictions()


if __name__ == "__main__":
    main()
