from groq import Groq
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

MODEL = "llama-3.3-70b-versatile"


# ==========================
# AI Dataset Analyzer
# ==========================

def ai_analyze_dataset(df):
    schema = df.dtypes.astype(str).to_dict()
    sample = df.head(3).to_json()

    prompt = f"""
    You are a data analyst.
    Dataset schema: {json.dumps(schema)}
    Sample rows: {sample}

    Return JSON ONLY. No explanation. No markdown.
    Return exactly this structure:
    {{
        "numeric_cols": ["col1", "col2"],
        "categorical_cols": ["col3"],
        "date_cols": ["col4"],
        "recommended_charts": [
            "Bar Chart - col3 vs col1",
            "Histogram - col2 distribution"
        ]
    }}
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


# ==========================
# AI Chart Recommender
# ==========================

def ai_recommend_charts(df, numeric_cols, categorical_cols, date_cols):
    prompt = f"""
    You are a data visualization expert.
    Numeric columns: {numeric_cols}
    Categorical columns: {categorical_cols}
    Date columns: {date_cols}
    Dataset shape: {df.shape}

    Recommend the 5 best charts for business insights.
    Return JSON ONLY. No explanation. No markdown.
    Return exactly this structure:
    {{
        "charts": [
            {{
                "type": "Bar Chart",
                "x": "column_name",
                "y": "column_name",
                "reason": "one sentence why"
            }}
        ]
    }}
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


# ==========================
# AI Question Answerer
# ==========================

def ai_answer_question(question, df):
    stats = df.describe().to_json()
    columns = df.columns.tolist()
    shape = df.shape

    prompt = f"""
    You are a data analyst assistant.
    Dataset columns: {columns}
    Dataset shape: {shape}
    Dataset statistics: {stats}

    User question: {question}

    Answer the question using the data above.
    Be specific with numbers and column names.
    Keep answer under 3 sentences.
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()


# ==========================
# AI Insight Generator
# ==========================

def ai_generate_insights(df, numeric_cols, categorical_cols):
    stats = df[numeric_cols].describe().to_json() \
            if numeric_cols else "{}"

    top_values = {}
    for col in categorical_cols:
        try:
            top_values[col] = df[col].value_counts().head(3).to_dict()
        except:
            pass

    prompt = f"""
    You are a business intelligence analyst.
    Numeric statistics: {stats}
    Top categorical values: {json.dumps(top_values)}
    Numeric columns: {numeric_cols}
    Categorical columns: {categorical_cols}

    Generate 5 key business insights from this data.
    Each insight should be one clear sentence.
    Start each with an emoji like 📈 📊 🔍 💡 ⚠️
    Return as plain text, one insight per line.
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )
    return response.choices[0].message.content.strip()


# ==========================
# AI Executive Summary
# ==========================

def ai_executive_summary(df, numeric_cols, categorical_cols, date_cols):
    stats = df[numeric_cols].describe().to_json() \
            if numeric_cols else "{}"

    prompt = f"""
    You are a senior business analyst writing for executives.
    Dataset shape: {df.shape}
    Numeric columns: {numeric_cols}
    Categorical columns: {categorical_cols}
    Date columns: {date_cols}
    Key statistics: {stats}

    Write a professional 5-sentence executive summary.
    Cover: dataset overview, key metrics, trends,
    notable findings, and recommended actions.
    Write in clear business language.
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )
    return response.choices[0].message.content.strip()