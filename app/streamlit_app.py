
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Feedback Intelligence",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.metric-card {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #ddd;
}

.title-text {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: gray;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    "<p class='title-text'>🤖 Customer Feedback Intelligence System</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>AI-Powered Sentiment Analysis, Intent Detection & Ticket Routing</p>",
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# MODEL PERFORMANCE SECTION
# --------------------------------------------------

st.subheader("📈 Model Performance")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Accuracy",
        "91%"
    )

with m2:
    st.metric(
        "Precision",
        "90%"
    )

with m3:
    st.metric(
        "Recall",
        "92%"
    )

with m4:
    st.metric(
        "F1 Score",
        "91%"
    )

st.divider()

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("📝 Customer Feedback")

text = st.text_area(
    "Enter customer feedback",
    height=180,
    placeholder="Example: The app crashes every time I upload a file..."
)

# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

if st.button(
    "🚀 Analyze Feedback",
    use_container_width=True
):

    if len(text.strip()) == 0:

        st.warning(
            "Please enter customer feedback."
        )

    else:

        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={
                    "text": text
                }
            )

            result = response.json()

            st.session_state.history.append(
                {
                    "Sentiment":
                        result["sentiment"],

                    "Intent":
                        result["intent"],

                    "Confidence":
                        result["confidence"],

                    "Assigned Team":
                        result["assigned_team"],

                    "Priority":
                        result["priority"]
                }
            )

            st.divider()

            # --------------------------------------
            # SENTIMENT DISPLAY
            # --------------------------------------

            if result["sentiment"] == "Positive":

                st.success(
                    f"😊 Sentiment: {result['sentiment']}"
                )

            else:

                st.error(
                    f"😡 Sentiment: {result['sentiment']}"
                )

            # --------------------------------------
            # KPI CARDS
            # --------------------------------------

            st.subheader(
                "📊 Prediction Summary"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Intent",
                    result["intent"]
                )

            with c2:

                st.metric(
                    "Confidence",
                    f"{result['confidence']}%"
                )

            with c3:

                st.metric(
                    "Priority",
                    result["priority"]
                )

            with c4:

                st.metric(
                    "Assigned Team",
                    result["assigned_team"]
                )

            st.divider()

            # --------------------------------------
            # PROBABILITY CHART
            # --------------------------------------

            st.subheader(
                "📉 Sentiment Probability Distribution"
            )

            fig = go.Figure()

            fig.add_bar(
                x=[
                    "Negative",
                    "Positive"
                ],
                y=[
                    result["negative_prob"],
                    result["positive_prob"]
                ],
                text=[
                    f"{result['negative_prob']}%",
                    f"{result['positive_prob']}%"
                ],
                textposition="auto"
            )

            fig.update_layout(
                title="Model Confidence Distribution",
                height=450
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.divider()

            # --------------------------------------
            # ROUTING
            # --------------------------------------

            st.subheader(
                "🎯 Ticket Routing"
            )

            st.info(
                f"This ticket should be routed to: "
                f"{result['assigned_team']}"
            )

            st.divider()

            # --------------------------------------
            # RAW OUTPUT
            # --------------------------------------

            with st.expander(
                "🔍 View Raw Prediction JSON"
            ):

                st.json(result)

        except Exception as e:

            st.error(
                f"Error: {e}"
            )

# --------------------------------------------------
# HISTORY SECTION
# --------------------------------------------------

if len(st.session_state.history) > 0:

    st.divider()

    st.subheader(
        "📜 Prediction History"
    )

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

    csv = history_df.to_csv(
        index=False
    )

    st.download_button(
        "⬇ Download Prediction History",
        csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built with RoBERTa, FastAPI, Streamlit, PyTorch and Hugging Face Transformers"
)

