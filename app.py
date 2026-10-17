import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from ai_engine import (
    ai_analyze_dataset,
    ai_recommend_charts,
    ai_answer_question,
    ai_generate_insights,
    ai_executive_summary
)

st.set_page_config(page_title="GenBI AI", layout="wide")

st.title("🤖 GenBI - AI Dashboard Generator")
st.markdown("Upload any dataset and AI will automatically analyse, clean, and visualise it for you.")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx", "xls", "json", "parquet", "txt"]
)

if uploaded_file:

    # ==========================
    # Load Dataset
    # ==========================

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)

    elif uploaded_file.name.endswith(".json"):
        df = pd.read_json(uploaded_file)

    elif uploaded_file.name.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)

    elif uploaded_file.name.endswith(".txt"):
        df = pd.read_csv(uploaded_file, sep=None, engine="python")

    st.success(f"✅ Loaded Successfully: {uploaded_file.name}")

    # ==========================
    # Dataset Preview
    # ==========================

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ==========================
    # Dataset Summary
    # ==========================

    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    # ==========================
    # AI Column Analysis
    # ==========================

    st.subheader("🤖 AI Column Analysis")

    with st.spinner("AI is analysing your dataset..."):
        try:
            analysis = ai_analyze_dataset(df)
            numeric_cols = analysis.get("numeric_cols", [])
            categorical_cols = analysis.get("categorical_cols", [])
            date_cols = analysis.get("date_cols", [])
            recommended = analysis.get("recommended_charts", [])
        except Exception as e:
            st.warning(f"AI analysis failed, using auto-detection. Error: {e}")
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
            recommended = []

    with st.expander("Numeric Columns"):
        st.write(numeric_cols)

    with st.expander("Categorical Columns"):
        st.write(categorical_cols)

    with st.expander("Date Columns"):
        st.write(date_cols)

    # ==========================
    # AI Recommended Charts
    # ==========================

    st.subheader("🤖 AI Recommended Charts")

    for chart in recommended:
        st.success(f"📊 {chart}")

    # ==========================
    # Dashboard Filters
    # ==========================

    st.subheader("Dashboard Filters")

    filtered_df = df.copy()

    for col in categorical_cols:
        unique_values = df[col].dropna().unique().tolist()
        selected_values = st.multiselect(
            f"Select {col}",
            unique_values,
            default=unique_values
        )
        filtered_df = filtered_df[
            filtered_df[col].isin(selected_values)
        ]

    df = filtered_df

    st.success(f"Filtered Dataset: {df.shape[0]} rows remaining")

    # ==========================
    # Data Cleaning
    # ==========================

    st.subheader("Data Cleaning Report")

    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates()

    empty_cols = df.columns[df.isnull().all()].tolist()
    df = df.dropna(axis=1, how="all")

    missing_before = int(df.isnull().sum().sum())

    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object", "category"]).columns:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

    missing_after = int(df.isnull().sum().sum())

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Duplicates Removed", duplicates_removed)

    with col2:
        st.metric("Empty Columns Removed", len(empty_cols))

    with col3:
        st.metric("Missing Before", missing_before)

    with col4:
        st.metric("Missing After", missing_after)

    st.write("Cleaned Dataset Shape:", df.shape)

    # ==========================
    # Cleaned Dataset Preview
    # ==========================

    st.subheader("Cleaned Dataset Preview")
    st.dataframe(df.head())

    # ==========================
    # Refresh Columns After Cleaning
    # ==========================

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # ==========================
    # Auto KPIs
    # ==========================

    st.subheader("Auto Generated KPIs")

    if len(numeric_cols) == 0:
        st.warning("No numeric columns found.")
    else:
        for col in numeric_cols:
            st.markdown(f"### {col}")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            with kpi1:
                st.metric("Total", round(float(df[col].sum()), 2))

            with kpi2:
                st.metric("Average", round(float(df[col].mean()), 2))

            with kpi3:
                st.metric("Maximum", round(float(df[col].max()), 2))

            with kpi4:
                st.metric("Minimum", round(float(df[col].min()), 2))

    # ==========================
    # Charts Section with Tabs
    # ==========================

    st.subheader("📊 Interactive Charts")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Histogram",
        "📈 Bar Chart",
        "🥧 Pie Chart",
        "📦 Box Plot",
        "🔵 Scatter Plot",
        "📉 Line Chart",
        "🔥 Heatmap"
    ])

    # --- Histogram Tab ---
    with tab1:
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found.")
        else:
            selected_col = st.selectbox(
                "Select column",
                numeric_cols,
                key="hist_col"
            )
            fig = px.histogram(
                df, x=selected_col,
                title=f"Distribution of {selected_col}"
            )
            st.plotly_chart(fig, use_container_width=True, key="hist_main")

            with st.expander("Show all histograms"):
                for idx, col in enumerate(numeric_cols):
                    fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True, key=f"hist_all_{idx}")

    # --- Bar Chart Tab ---
    with tab2:
        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            st.warning("Need both categorical and numeric columns.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                cat_col = st.selectbox("Select category column", categorical_cols, key="bar_cat")
            with col_b:
                num_col = st.selectbox("Select numeric column", numeric_cols, key="bar_num")

            try:
                chart_data = df.groupby(cat_col)[num_col].sum().reset_index()
                fig = px.bar(chart_data, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}")
                st.plotly_chart(fig, use_container_width=True, key="bar_main")
            except:
                st.warning("Could not generate bar chart.")

            with st.expander("Show all bar charts"):
                idx = 0
                for cat in categorical_cols:
                    for num in numeric_cols:
                        try:
                            chart_data = df.groupby(cat)[num].sum().reset_index()
                            fig = px.bar(chart_data, x=cat, y=num, title=f"{num} by {cat}")
                            st.plotly_chart(fig, use_container_width=True, key=f"bar_all_{idx}")
                            idx += 1
                        except:
                            pass

    # --- Pie Chart Tab ---
    with tab3:
        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            st.warning("Need both categorical and numeric columns.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                cat_col = st.selectbox("Select category column", categorical_cols, key="pie_cat")
            with col_b:
                num_col = st.selectbox("Select numeric column", numeric_cols, key="pie_num")

            try:
                pie_data = df.groupby(cat_col)[num_col].sum().reset_index().head(10)
                fig = px.pie(pie_data, names=cat_col, values=num_col, title=f"{num_col} by {cat_col}")
                st.plotly_chart(fig, use_container_width=True, key="pie_main")
            except:
                st.warning("Could not generate pie chart.")

            with st.expander("Show all pie charts"):
                idx = 0
                for cat in categorical_cols:
                    for num in numeric_cols:
                        try:
                            pie_data = df.groupby(cat)[num].sum().reset_index().head(10)
                            fig = px.pie(pie_data, names=cat, values=num, title=f"{num} by {cat}")
                            st.plotly_chart(fig, use_container_width=True, key=f"pie_all_{idx}")
                            idx += 1
                        except:
                            pass

    # --- Box Plot Tab ---
    with tab4:
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found.")
        else:
            selected_col = st.selectbox(
                "Select column",
                numeric_cols,
                key="box_col"
            )
            fig = px.box(df, y=selected_col, title=f"Box Plot - {selected_col}")
            st.plotly_chart(fig, use_container_width=True, key="box_main")

            with st.expander("Show all box plots"):
                for idx, col in enumerate(numeric_cols):
                    fig = px.box(df, y=col, title=f"Box Plot - {col}")
                    st.plotly_chart(fig, use_container_width=True, key=f"box_all_{idx}")

    # --- Scatter Plot Tab ---
    with tab5:
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                x_col = st.selectbox("Select X column", numeric_cols, key="scatter_x")
            with col_b:
                y_col = st.selectbox("Select Y column", numeric_cols, key="scatter_y")

            fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True, key="scatter_main")

            with st.expander("Show all scatter plots"):
                idx = 0
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        fig = px.scatter(
                            df,
                            x=numeric_cols[i],
                            y=numeric_cols[j],
                            title=f"{numeric_cols[i]} vs {numeric_cols[j]}"
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"scatter_all_{idx}")
                        idx += 1

    # --- Line Chart Tab ---
    with tab6:
        if len(date_cols) == 0:
            st.info("No date columns detected. Select any column as X axis.")
            col_a, col_b = st.columns(2)
            with col_a:
                x_col = st.selectbox("Select X column", df.columns.tolist(), key="line_x")
            with col_b:
                y_col = st.selectbox("Select Y column", numeric_cols, key="line_y")
            try:
                fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                st.plotly_chart(fig, use_container_width=True, key="line_main_nodate")
            except:
                st.warning("Could not generate line chart.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                date_col = st.selectbox("Select date column", date_cols, key="line_date")
            with col_b:
                num_col = st.selectbox("Select numeric column", numeric_cols, key="line_num")
            try:
                fig = px.line(df, x=date_col, y=num_col, title=f"{num_col} over {date_col}")
                st.plotly_chart(fig, use_container_width=True, key="line_main")
            except:
                st.warning("Could not generate line chart.")

            with st.expander("Show all line charts"):
                idx = 0
                for date in date_cols:
                    for num in numeric_cols:
                        try:
                            fig = px.line(df, x=date, y=num, title=f"{num} over {date}")
                            st.plotly_chart(fig, use_container_width=True, key=f"line_all_{idx}")
                            idx += 1
                        except:
                            pass

    # --- Heatmap Tab ---
    with tab7:
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for a heatmap.")
        else:
            corr = df[numeric_cols].corr()
            fig = ff.create_annotated_heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.index),
                annotation_text=round(corr, 2).values,
                showscale=True
            )
            st.plotly_chart(fig, use_container_width=True, key="heatmap_main")

    # ==========================
    # AI Insights
    # ==========================

    st.subheader("🤖 AI Generated Insights")

    with st.spinner("AI is generating insights..."):
        try:
            insights = ai_generate_insights(df, numeric_cols, categorical_cols)
            st.write(insights)
        except Exception as e:
            st.warning(f"Could not generate insights. Error: {e}")

    # ==========================
    # AI Q&A
    # ==========================

    st.subheader("🤖 Ask Questions About Your Data")

    question = st.text_input("Ask anything about your data")

    if question:
        with st.spinner("AI is thinking..."):
            try:
                answer = ai_answer_question(question, df)
                st.success(answer)
            except Exception as e:
                st.warning(f"Could not answer question. Error: {e}")

    # ==========================
    # AI Executive Summary
    # ==========================

    st.subheader("🤖 AI Executive Summary")

    with st.spinner("AI is writing executive summary..."):
        try:
            summary = ai_executive_summary(
                df, numeric_cols, categorical_cols, date_cols
            )
            st.write(summary)
        except Exception as e:
            st.warning(f"Could not generate summary. Error: {e}")