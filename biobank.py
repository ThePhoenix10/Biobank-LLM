import io
import re
import time

import duckdb
import pandas as pd
from openai import OpenAI, RateLimitError

MODEL = "gpt-4o"
TABLE_NAME = "inventory"
MAX_RETRIES = 2
LOW_CARDINALITY_LIMIT = 60
SAMPLE_VALUES_SHOWN = 8
RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_BASE_DELAY_SECONDS = 2

FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|ATTACH|COPY|EXPORT|PRAGMA|GRANT)\b",
    re.IGNORECASE,
)

PII_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"),
    re.compile(r"\bmy name is\b", re.IGNORECASE),
    re.compile(r"\b(dob|date of birth)\b", re.IGNORECASE),
    re.compile(r"\bmy (specimen|record|sample|data)\b", re.IGNORECASE),
    re.compile(r"\bssn\b", re.IGNORECASE),
]


def contains_pii(question: str) -> bool:
    return any(p.search(question) for p in PII_PATTERNS)

SYSTEM_PROMPT_TEMPLATE = """You are a SQL generator for a DuckDB database. You translate a user's \
natural-language question into a single, valid, READ-ONLY SQL query.

{schema}

Rules:
- If the question is unrelated to this biobank data, nonsensical, or cannot be \
answered from the columns above, output exactly this and nothing else: NOT_APPLICABLE
- Output ONLY the SQL query. No explanation, no markdown fences, no commentary.
- Only write SELECT statements. Never write INSERT/UPDATE/DELETE/DROP/ALTER/etc.
- Column names contain spaces and slashes — always wrap them in double quotes, \
exactly as spelled above, e.g. "Patient Deceased Y/N".
- Use only the exact column values listed above (the "allowed values"). Do not \
invent categories that aren't in that list.
- Treat '-', 'Unknown', 'N/R', 'Not Recorded', and '*Blank' as missing/unknown \
values, not as real categories, unless the user specifically asks about missing data.
- "Patient Age" is stored as text because some entries are '90+'; cast with \
TRY_CAST("Patient Age" AS INTEGER) when doing numeric comparisons or averages, \
and it will exclude non-numeric ages like '90+' from that calculation. \
IMPORTANT: for "over/older than/above N" comparisons where N < 90, you must \
also include the '90+' bucket explicitly, since it's always true that someone \
in the '90+' bucket is older than any N below 90. Write it as: \
(TRY_CAST("Patient Age" AS INTEGER) > N OR "Patient Age" = '90+'). This does \
not apply to "under/younger than" comparisons, since '90+' would never qualify.
- "in <year>" (e.g. "in 2018", "in 2009") refers to "Path Date Year" — the \
year the specimen/pathology record is dated — NOT "Death Year", even when \
the question also mentions deceased patients. Only use "Death Year" when the \
question explicitly asks about when someone died (e.g. "died in 2018", \
"death year is 2018").
- The word "matched" before a diagnosis or disease name (e.g. "matched \
Chronic Lymphocytic Leukemia cases") is almost always a filler word, not a \
reference to "Specimen Type" = 'Matched Normal Specimen' or "Diagnosis \
Type" = 'Matched Normal Specimen'. Only interpret "matched" as filtering on \
that value when the question is specifically about matched normal/control \
specimens.
- Terminology for what to count:
  - "patient(s)" -> COUNT(DISTINCT "Unique Patient ID")
  - "case(s)"    -> COUNT(DISTINCT "Catalog No")
  - "specimen(s)" / "record(s)" (or no qualifier given) -> COUNT(*)
- When the question names a body part, organ, or tissue (e.g. "breast", \
"lung", "kidney", "skin", "colorectal", "head and neck"), filter on \
"Primary Site" — not "Diagnosis Type" — unless the question explicitly asks \
about a diagnosis/carcinoma/disease type instead of an anatomical site.
- Known multi-value anatomical groupings (use these exact "Primary Site" \
value lists when the question uses these group names):
  - "colorectal" -> "Primary Site" IN ('Colon', 'Rectum')
  - "head and neck" -> "Primary Site" IN ('Larynx', 'Oral Cavity', \
'Salivary Gland', 'Tongue', 'Tonsil')  [note: this group does NOT include \
Thyroid Gland]
"""


def load_excel_to_duckdb(file_bytes: bytes):
    df = pd.read_excel(io.BytesIO(file_bytes))
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).replace("nan", None)
    con = duckdb.connect(database=":memory:")
    con.register("df_view", df)
    con.execute(f'CREATE TABLE "{TABLE_NAME}" AS SELECT * FROM df_view')
    n_rows = con.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0]
    return con, n_rows, len(df.columns)


def build_schema_description(con) -> str:
    cols = con.execute(f'DESCRIBE "{TABLE_NAME}"').fetchdf()
    lines = [f'Table "{TABLE_NAME}" columns:']
    for _, row in cols.iterrows():
        col, dtype = row["column_name"], row["column_type"]
        n_distinct = con.execute(f'SELECT COUNT(DISTINCT "{col}") FROM "{TABLE_NAME}"').fetchone()[0]
        if 0 < n_distinct <= LOW_CARDINALITY_LIMIT:
            vals = con.execute(
                f'SELECT DISTINCT "{col}" FROM "{TABLE_NAME}" WHERE "{col}" IS NOT NULL ORDER BY 1'
            ).fetchdf()[col].tolist()
            lines.append(f'  - "{col}" ({dtype}) — allowed values: {vals}')
        else:
            samples = con.execute(
                f'SELECT DISTINCT "{col}" FROM "{TABLE_NAME}" WHERE "{col}" IS NOT NULL '
                f'ORDER BY 1 LIMIT {SAMPLE_VALUES_SHOWN}'
            ).fetchdf()[col].tolist()
            lines.append(f'  - "{col}" ({dtype}) — example values: {samples}')
    return "\n".join(lines)


def clean_sql(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```sql\s*|^```\s*|```$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip().rstrip(";")


def _call_with_rate_limit_retry(fn):
    delay = RATE_LIMIT_BASE_DELAY_SECONDS
    last_error = None
    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return fn()
        except RateLimitError as e:
            last_error = e
            if attempt >= RATE_LIMIT_MAX_RETRIES:
                raise
            time.sleep(delay)
            delay *= 2
    raise last_error


def generate_sql(client: OpenAI, schema: str, question: str, history_error: str | None = None) -> str:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(schema=schema)
    user_content = question
    if history_error:
        user_content = f"{question}\n\nYour previous SQL attempt failed with this error, fix it:\n{history_error}"

    def _call():
        return client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )

    resp = _call_with_rate_limit_retry(_call)
    return clean_sql(resp.choices[0].message.content)


def is_safe_select(sql: str) -> bool:
    if FORBIDDEN.search(sql):
        return False
    if not sql.strip().upper().startswith("SELECT"):
        return False
    if ";" in sql:
        return False
    return True


def summarize_result(client: OpenAI, question: str, df: pd.DataFrame) -> str:
    preview = df.head(20).to_csv(index=False)

    def _call():
        return client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question in 1-2 concise sentences "
                    "using only the data given. Do not add outside knowledge.",
                },
                {"role": "user", "content": f"Question: {question}\n\nQuery result (CSV):\n{preview}"},
            ],
        )

    resp = _call_with_rate_limit_retry(_call)
    return resp.choices[0].message.content.strip()


def ask(con, client: OpenAI, schema: str, question: str) -> dict:
    if contains_pii(question):
        return {"status": "pii_blocked"}

    error = None
    sql = None
    for attempt in range(MAX_RETRIES + 1):
        sql = generate_sql(client, schema, question, history_error=error)

        if sql.strip().upper() == "NOT_APPLICABLE":
            return {"status": "not_applicable"}

        if not is_safe_select(sql):
            return {"status": "blocked", "sql": sql}

        try:
            df = con.execute(sql).fetchdf()
            break
        except Exception as e:
            error = str(e)
            if attempt < MAX_RETRIES:
                continue
            return {"status": "failed", "sql": sql, "error": error, "attempts": MAX_RETRIES + 1}

    result = {"status": "ok", "sql": sql, "df": df}
    try:
        result["summary"] = summarize_result(client, question, df)
    except Exception as e:
        result["summary_error"] = str(e)
    return result
