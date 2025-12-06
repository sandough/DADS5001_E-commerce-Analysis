import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from groq import Groq  # ✅ ใช้ Groq สำหรับ AI Insight

# ----------------- Page config -----------------
st.set_page_config(page_title="Customer Analysis", page_icon="📊", layout="wide")
st.title("📊 การวิเคราะห์ลูกค้า (Customer Analysis)")

# ----------------- Groq API Key -----------------
groq_api_key = "MY_API_KEY"

@st.cache_resource
def get_groq_client(api_key: str):
    return Groq(api_key=api_key)

# ----------------- AI Prompt Builders -----------------
def build_country_demand_insight(country_df: pd.DataFrame) -> str:
    # สรุปยอดรวมรายประเทศ (เฉพาะ top 15 ที่ใช้ในกราฟ)
    summary = (
        country_df.groupby("Country")
        .agg(
            TotalFrequency=("Frequency", "sum"),
            TotalQuantity=("TotalQuantity", "sum"),
            ActiveMonths=("Month", "nunique"),
        )
        .reset_index()
        .sort_values("TotalQuantity", ascending=False)
    )

    lines = []
    for row in summary.itertuples():
        lines.append(
            f"- {row.Country}: คำสั่งซื้อ {row.TotalFrequency:,} ครั้ง, "
            f"ปริมาณ {row.TotalQuantity:,} ชิ้น, ใช้งาน {row.ActiveMonths} เดือน"
        )

    text = "\n".join(lines)

    prompt = f"""
คุณคือ Data Analyst ช่วยวิเคราะห์ "ความต้องการของลูกค้าแบ่งตามประเทศรายเดือน" ด้านล่างนี้

ข้อมูลสรุป (เฉพาะประเทศ Top ที่ใช้ในกราฟ):
{text}

กรุณาเขียน insight เป็น bullet point ภาษาไทย (ไม่เกิน 6 ข้อ):
- ประเทศใดมีความต้องการสูงสุด และต่างจากประเทศอื่นอย่างไร
- มี pattern ด้านฤดูกาล / เดือนที่โดดเด่นหรือไม่ (เช่น เดือนไหนพุ่งขึ้น/ตกลงชัดเจน)
- กลุ่มประเทศที่พฤติกรรมคล้ายกันมีประเทศใดบ้าง
- ข้อเสนอเชิงธุรกิจ 1–2 ข้อ เช่น ควรโฟกัสประเทศไหนในช่วงเดือนไหน

ตอบเป็น bullet point เท่านั้น
"""
    return prompt


def build_region_demand_insight(region_df: pd.DataFrame) -> str:
    summary = (
        region_df.groupby("Region")
        .agg(
            TotalFrequency=("Frequency", "sum"),
            TotalQuantity=("TotalQuantity", "sum"),
            ActiveMonths=("Month", "nunique"),
        )
        .reset_index()
        .sort_values("TotalQuantity", ascending=False)
    )

    lines = []
    for row in summary.itertuples():
        lines.append(
            f"- {row.Region}: คำสั่งซื้อ {row.TotalFrequency:,} ครั้ง, "
            f"ปริมาณ {row.TotalQuantity:,} ชิ้น, ใช้งาน {row.ActiveMonths} เดือน"
        )
    text = "\n".join(lines)

    prompt = f"""
ช่วยวิเคราะห์ "ความต้องการของลูกค้าแบ่งตามภูมิภาครายเดือน" จากข้อมูลสรุปนี้:

{text}

กรุณาเขียน insight เป็น bullet point ภาษาไทย:
- เปรียบเทียบภูมิภาคที่โดดเด่นด้านจำนวนคำสั่งซื้อและปริมาณ
- มีภูมิภาคไหนที่เติบโต/ชะลอตัวตามเดือน (seasonality คร่าวๆ)
- ภูมิภาคใดควรโฟกัสเป็น priority และเพราะอะไร
- ข้อเสนอแนะ 1–2 ข้อสำหรับการจัดสรรสต็อกหรือทำแคมเปญ

ตอบเป็น bullet point เท่านั้น
"""
    return prompt


def build_aov_group_insight(continent_summary: pd.DataFrame) -> str:
    lines = []
    for row in continent_summary.sort_values("AOV", ascending=False).itertuples():
        lines.append(f"- {row.Group}: AOV เฉลี่ย £{row.AOV:,.2f}")
    text = "\n".join(lines)

    prompt = f"""
ข้อมูลนี้คือค่าเฉลี่ยมูลค่าคำสั่งซื้อ (AOV) รายทวีป:

{text}

ช่วยสรุป insight เป็น bullet point ภาษาไทย:
- ทวีปใดมี AOV สูง/ต่ำ และช่วงห่างประมาณเท่าไร
- มี pattern ที่บ่งบอก segment ลูกค้าแต่ละทวีปหรือไม่
- ไอเดียกลยุทธ์ราคา / bundle / premium market ต่อทวีป

ตอบเป็น bullet point เท่านั้น
"""
    return prompt


def build_kpi_retention_insight(
    total_purchases,
    total_customers,
    total_quantity,
    cancel_count,
    cancel_sum,
    cancel_aov,
    cancel_ratio,
    retention_df: pd.DataFrame,
) -> str:
    retention_summary = ""
    if len(retention_df) > 0:
        avg_months = retention_df["MonthsActive"].mean()
        max_months = retention_df["MonthsActive"].max()
        retention_summary = (
            f"- ลูกค้าที่กลับมาซื้อซ้ำ: {len(retention_df):,} ราย\n"
            f"- จำนวนเดือนเฉลี่ยที่กลับมาซื้อซ้ำ: {avg_months:.1f} เดือน (สูงสุด {max_months} เดือน)"
        )

    prompt = f"""
สรุปตัวชี้วัดหลักของธุรกิจ:

- คำสั่งซื้อรวม: {total_purchases:,} รายการ
- ลูกค้ารวม: {total_customers:,} ราย
- จำนวนสินค้าที่ขายได้: {total_quantity:,.0f} ชิ้น

สถานะคำสั่งซื้อที่ยกเลิก:
- จำนวนคำสั่งซื้อที่ยกเลิก: {cancel_count:,} รายการ
- มูลค่ารวมที่ยกเลิก: £{cancel_sum:,.2f}
- มูลค่าเฉลี่ยต่อคำสั่งซื้อที่ยกเลิก: £{cancel_aov:,.2f}
- สัดส่วนคำสั่งซื้อที่ยกเลิก: {cancel_ratio:.2f}%

Retention:
{retention_summary}

ช่วยสรุป insight เป็น bullet point ภาษาไทย:
- มองภาพรวมสุขภาพธุรกิจจากตัวเลขเหล่านี้
- ประเมินความน่ากังวลของสัดส่วนการยกเลิก และควรโฟกัสแก้ที่จุดใด
- ความแข็งแรงของฐานลูกค้าซื้อซ้ำ และโอกาสทำ CRM / Loyalty
- ข้อเสนอเชิงกลยุทธ์ 2–3 ข้อ

ตอบเป็น bullet point เท่านั้น
"""
    return prompt


def build_pareto_insight(summary_df: pd.DataFrame) -> str:
    lines = []
    for row in summary_df.itertuples():
        lines.append(
            f"- {row.Index}. {row.Category}: ยอดขาย £{row.TotalSales:,.2f} "
            f"({row.SalesPercent:.2f}%) | จำนวนสินค้า {row.ProductCount:,.0f} รายการ "
            f"({row.ProductPercent:.2f}%)"
        )
    text = "\n".join(lines)

    prompt = f"""
ข้อมูลนี้คือผล Pareto Analysis แยกตามหมวดสินค้า (เฉพาะสินค้าที่สร้าง 80% ของยอดขาย):

{text}

ช่วยวิเคราะห์เป็น bullet point ภาษาไทย:
- หมวดใดสร้างรายได้หลัก และสัดส่วนเมื่อเทียบกับจำนวนสินค้า
- มีหมวด "ดาวเด่น" ที่ยอดขายสูงแต่จำนวน SKU ไม่มากหรือไม่
- มีหมวดที่ SKU เยอะแต่ยอดขายไม่เด่น (อาจเป็น candidate สำหรับ rationalization)
- เสนอแนวคิด 2–3 ข้อสำหรับการจัด assortment / stock / campaign

ตอบเป็น bullet point เท่านั้น
"""
    return prompt

# ----------------- Load data -----------------
@st.cache_data(ttl=60)
def load_data():
    url = 'https://docs.google.com/spreadsheets/d/12vD8wGU1HvXxpdFowsO7pgcXucI30Ei-gN2hRZEkL6s/export?format=csv'
    df = pd.read_csv(url)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    df['Month'] = df['InvoiceDate'].dt.month
    df['MonthName'] = df['InvoiceDate'].dt.strftime('%b')
    return df

df = load_data()

# ----------------- Country grouping -----------------
asian_countries = ['Japan', 'Singapore', 'Hong Kong', 'Korea', 'China', 'Thailand',
                   'Malaysia', 'Indonesia', 'Philippines', 'Vietnam', 'India', 'UAE', 'Saudi Arabia']
eu_countries = ['United Kingdom', 'Germany', 'France', 'Spain', 'Italy', 'Netherlands',
                'Belgium', 'Switzerland', 'Portugal', 'Sweden', 'Norway', 'Denmark',
                'Finland', 'Austria', 'Poland', 'Greece', 'Ireland', 'Czech Republic']

def classify_region(country):
    if country in asian_countries:
        return 'Asian Countries'
    elif country in eu_countries:
        return 'EU Countries'
    else:
        return 'Other Regions'

df['Region'] = df['Country'].apply(classify_region)

# DuckDB base
con = duckdb.connect(':memory:')
con.register('df_table', df)

# ====================================================
# SECTION 1: Individual Countries
# ====================================================
st.header("🌍 ความต้องการของลูกค้าแบ่งตามประเทศ")

query_country = """
SELECT 
    Country,
    Month,
    MonthName,
    COUNT(DISTINCT InvoiceNo) as Frequency,
    SUM(Quantity) as TotalQuantity
FROM df_table
WHERE Quantity > 0
GROUP BY Country, Month, MonthName
ORDER BY Country, Month
"""
country_data = con.execute(query_country).df()

tab1, tab2 = st.tabs(["ความถี่ในการซื้อสินค้า", "ปริมาณคำสั่งซื้อ"])

with tab1:
    st.subheader("ความถี่ในการซื้อสินค้าของแต่ละประเทศแบ่งตามช่วงเวลา")

    top_countries = con.execute("""
        SELECT Country, SUM(Quantity) as Total
        FROM df_table
        WHERE Quantity > 0
        GROUP BY Country
        ORDER BY Total DESC
        LIMIT 15
    """).df()

    country_data_filtered = country_data[country_data['Country'].isin(top_countries['Country'])]

    fig_line = px.line(
        country_data_filtered,
        x='Month',
        y='Frequency',
        color='Country',
        markers=True,
        title='Top 15 ประเทศที่มีความถี่ในการซื้อสินค้ามากที่สุดแบ่งตามเดือน',
        labels={'Frequency': 'จำนวนคำสั่งซื้อ', 'Month': 'เดือน'}
    )
    fig_line.update_layout(height=600, hovermode='x unified',
                           xaxis=dict(tickmode='linear', dtick=1))
    st.plotly_chart(fig_line, use_container_width=True)

with tab2:
    st.subheader("Heatmap แสดงปริมาณคำสั่งซื้อของแต่ละประเทศแบ่งตามช่วงเวลา")

    heatmap_data = country_data.pivot_table(
        index='Country',
        columns='Month',
        values='TotalQuantity',
        fill_value=0
    )

    top_15_countries = heatmap_data.sum(axis=1).nlargest(15).index
    heatmap_data_filtered = heatmap_data.loc[top_15_countries]

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data_filtered.values,
        x=month_labels,
        y=heatmap_data_filtered.index,
        colorscale='YlOrRd',
        text=heatmap_data_filtered.values,
        texttemplate='%{text:.0f}',
        textfont={"size": 12},
        colorbar=dict(title="Quantity")
    ))
    fig_heatmap.update_layout(
        title='Top 15 ประเทศที่มีปริมาณคำสั่งซื้อมากที่สุดแบ่งตามเดือน',
        xaxis_title='เดือน',
        yaxis_title='ประเทศ',
        height=650,
        yaxis=dict(autorange='reversed')
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ---- AI Insight: Section 1 ----
st.subheader("🤖 AI Insights: ความต้องการลูกค้าแบ่งตามประเทศ")
mode_country_ai = st.radio(
    "โหมดการแสดงผล (ความต้องการรายประเทศ)",
    ["แสดงกราฟอย่างเดียว", "ให้ AI วิเคราะห์ส่วนนี้"],
    horizontal=True,
    key="mode_country_ai",
)
if mode_country_ai == "ให้ AI วิเคราะห์ส่วนนี้":
    with st.spinner("AI กำลังวิเคราะห์ความต้องการรายประเทศ..."):
        client = get_groq_client(groq_api_key)
        prompt = build_country_demand_insight(country_data_filtered)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "คุณเป็นผู้เชี่ยวชาญด้านการวิเคราะห์ข้อมูลลูกค้าและ Demand"},
                {"role": "user", "content": prompt},
            ],
        )
        insight = completion.choices[0].message.content
    st.markdown(insight)

# ====================================================
# SECTION 2: Regional Groups
# ====================================================
st.header("🌏 ความต้องการของลูกค้าแบ่งตามภูมิภาค")

query_region = """
SELECT 
    Region,
    Month,
    MonthName,
    COUNT(DISTINCT InvoiceNo) as Frequency,
    SUM(Quantity) as TotalQuantity
FROM df_table
WHERE Quantity > 0
GROUP BY Region, Month, MonthName
ORDER BY Region, Month
"""
region_data = con.execute(query_region).df()

tab3, tab4 = st.tabs(["ความถี่ในการซื้อสินค้า", "ปริมาณคำสั่งซื้อ"])

with tab3:
    st.subheader("ความถี่ในการซื้อสินค้าของแต่ละภูมิภาคแบ่งตามช่วงเวลา")

    fig_region_line = px.line(
        region_data,
        x='Month',
        y='Frequency',
        color='Region',
        markers=True,
        title='เปรียบเทียบความถี่ในการซื้อสินค้าของแต่ละภูมิภาคแบ่งตามเดือน',
        labels={'Frequency': 'จำนวนคำสั่งซื้อ', 'Month': 'เดือน'}
    )
    fig_region_line.update_layout(
        height=500, hovermode='x unified',
        xaxis=dict(tickmode='linear', dtick=1)
    )
    st.plotly_chart(fig_region_line, use_container_width=True)

    st.subheader("📊 ปริมาณคำสั่งซื้อรวมของแต่ละภูมิภาคแบ่งตามช่วงเวลา")
    fig_quantity = px.line(
        region_data,
        x='Month',
        y='TotalQuantity',
        color='Region',
        markers=True,
        title='ปริมาณคำสั่งซื้อรวมของแต่ละภูมิภาคแบ่งตามเดือน',
        labels={'Month': 'เดือน', 'TotalQuantity': 'ปริมาณคำสั่งซื้อรวม'}
    )
    fig_quantity.update_layout(
        height=400,
        hovermode='x unified',
        xaxis=dict(tickmode='linear', dtick=1)
    )
    st.plotly_chart(fig_quantity, use_container_width=True)

with tab4:
    st.subheader("Heatmap แสดงปริมาณคำสั่งซื้อรวมของแต่ละภูมิภาคแบ่งตามช่วงเวลา")

    region_heatmap = region_data.pivot_table(
        index='Region',
        columns='Month',
        values='TotalQuantity',
        fill_value=0
    )

    fig_region_heatmap = go.Figure(data=go.Heatmap(
        z=region_heatmap.values,
        x=month_labels,
        y=region_heatmap.index,
        colorscale='Viridis',
        text=region_heatmap.values,
        texttemplate='%{text:.0f}',
        textfont={"size": 12},
        colorbar=dict(title="Quantity")
    ))
    fig_region_heatmap.update_layout(
        title='ปริมาณคำสั่งซื้อของแต่ละภูมิภาคแบ่งตามเดือน',
        xaxis_title='เดือน',
        yaxis_title='ภูมิภาค',
        height=400
    )
    st.plotly_chart(fig_region_heatmap, use_container_width=True)

# ---- AI Insight: Section 2 ----
st.subheader("🤖 AI Insights: ความต้องการลูกค้าแบ่งตามภูมิภาค")
mode_region_ai = st.radio(
    "โหมดการแสดงผล (ความต้องการรายภูมิภาค)",
    ["แสดงกราฟอย่างเดียว", "ให้ AI วิเคราะห์ส่วนนี้"],
    horizontal=True,
    key="mode_region_ai",
)
if mode_region_ai == "ให้ AI วิเคราะห์ส่วนนี้":
    with st.spinner("AI กำลังวิเคราะห์ความต้องการรายภูมิภาค..."):
        client = get_groq_client(groq_api_key)
        prompt = build_region_demand_insight(region_data)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "คุณเป็นผู้เชี่ยวชาญด้านการวิเคราะห์ Demand รายภูมิภาค"},
                {"role": "user", "content": prompt},
            ],
        )
    insight = completion.choices[0].message.content
    st.markdown(insight)

st.divider()

# ====================================================
# SECTION 3: AOV by Country / Continent
# ====================================================
aov_query = """
WITH cleaned AS (
    SELECT
        InvoiceNo,
        Country,
        SUM(Quantity * UnitPrice) AS InvoiceSales
    FROM df
    WHERE InvoiceNo NOT LIKE 'C%'  
    GROUP BY InvoiceNo, Country
)
SELECT
    Country,
    AVG(InvoiceSales) AS AOV
FROM cleaned
GROUP BY Country
ORDER BY AOV DESC;
"""
st.header("📊 E-commerce Analytics: AOV แบ่งตามประเทศและทวีป")

aov = duckdb.query(aov_query).to_df()

# -----------------------------
# Continent mapping แบบง่ายสำหรับ Online Retail
# -----------------------------
continent_mapping = {
    # Europe
    "United Kingdom": "Europe",
    "EIRE": "Europe",
    "Netherlands": "Europe",
    "Germany": "Europe",
    "France": "Europe",
    "Spain": "Europe",
    "Portugal": "Europe",
    "Belgium": "Europe",
    "Switzerland": "Europe",
    "Norway": "Europe",
    "Sweden": "Europe",
    "Finland": "Europe",
    "Italy": "Europe",
    "Austria": "Europe",
    "Denmark": "Europe",
    "Poland": "Europe",
    "Greece": "Europe",
    "Cyprus": "Europe",
    "Channel Islands": "Europe",
    "Iceland": "Europe",
    "Malta": "Europe",
    "Lithuania": "Europe",
    "Czech Republic": "Europe",
    "European Community" : "Europe",
    "Albania": "Europe",
    "Andorra": "Europe",
    "Belarus": "Europe",
    "Bosnia and Herzegovina": "Europe",
    "Bulgaria": "Europe",
    "Croatia": "Europe",
    "Estonia": "Europe",
    "Faroe Islands": "Europe",
    "Gibraltar": "Europe",
    "Guernsey": "Europe",
    "Holy See": "Europe",
    "Hungary": "Europe",
    "Ireland": "Europe",
    "Isle of Man": "Europe",
    "Jersey": "Europe",
    "Latvia": "Europe",
    "Liechtenstein": "Europe",
    "Luxembourg": "Europe",
    "Malta": "Europe",
    "Monaco": "Europe",
    "Montenegro": "Europe",
    "North Macedonia": "Europe",
    "Republic of Moldova": "Europe",
    "Romania": "Europe",
    "San Marino": "Europe",
    "Serbia": "Europe",
    "Slovakia": "Europe",
    "Slovenia": "Europe",
    "Ukraine": "Europe",
    "Kosovo": "Europe",

    # Asia / Middle East
    "Israel": "Asia",
    "Japan": "Asia",
    "Singapore": "Asia",
    "Hong Kong": "Asia",
    "Thailand": "Asia",
    "Korea": "Asia",
    "China": "Asia",
    "Saudi Arabia": "Asia",
    "United Arab Emirates": "Asia",
    "Lebanon": "Asia",
    "Bahrain" : "Asia",
    "Afghanistan": "Asia",
    "Armenia": "Asia",
    "Azerbaijan": "Asia",
    "Bangladesh": "Asia",
    "Bhutan": "Asia",
    "Brunei Darussalam": "Asia",
    "Cambodia": "Asia",
    "Georgia": "Asia",
    "India": "Asia",
    "Indonesia": "Asia",
    "Iran": "Asia",
    "Iraq": "Asia",
    "Jordan": "Asia",
    "Kazakhstan": "Asia",
    "Kuwait": "Asia",
    "Kyrgyzstan": "Asia",
    "Laos": "Asia",
    "Macao": "Asia",
    "Malaysia": "Asia",
    "Maldives": "Asia",
    "Mongolia": "Asia",
    "Myanmar": "Asia",
    "Nepal": "Asia",
    "Oman": "Asia",
    "Pakistan": "Asia",
    "Palestine, State of": "Asia",
    "Philippines": "Asia",
    "Qatar": "Asia",
    "Republic of Korea": "Asia",
    "Sri Lanka": "Asia",
    "Syrian Arab Republic": "Asia",
    "Tajikistan": "Asia",
    "Timor-Leste": "Asia",
    "Turkey": "Asia",
    "Turkmenistan": "Asia",
    "Uzbekistan": "Asia",
    "Viet Nam": "Asia",
    "Yemen": "Asia",
 
    # Oceania
    "Australia": "Oceania",
    "New Zealand": "Oceania",

    # Americas & Africa
    "USA": "Americas",
    "Brazil": "Americas",
    "Canada": "Americas",
    "Belize": "Americas",
    "Costa Rica": "Americas",
    "El Salvador": "Americas",
    "Guatemala": "Americas",
    "Honduras": "Americas",
    "Mexico": "Americas",
    "Nicaragua": "Americas",
    "Panama": "Americas",
    "Antigua and Barbuda": "Americas",
    "Bahamas": "Americas",
    "Barbados": "Americas",
    "Cuba": "Americas",
    "Dominica": "Americas",
    "Dominican Republic": "Americas",
    "Grenada": "Americas",
    "Haiti": "Americas",
    "Jamaica": "Americas",
    "Saint Kitts and Nevis": "Americas",
    "Saint Lucia": "Americas",
    "Saint Vincent and the Grenadines": "Americas",
    "Trinidad and Tobago": "Americas",
    "Argentina": "Americas",
    "Bolivia": "Americas",
    "Brazil": "Americas",
    "Chile": "Americas",
    "Colombia": "Americas",
    "Ecuador": "Americas",
    "Guyana": "Americas",
    "Paraguay": "Americas",
    "Peru": "Americas",
    "Suriname": "Americas",
    "Uruguay (Oriental Republic of)": "Americas",
    "Venezuela (Bolivarian Republic of)": "Americas",

    # Africa
    "Algeria": "Africa",
    "Angola": "Africa",
    "Benin": "Africa",
    "Botswana": "Africa",
    "Burkina Faso": "Africa",
    "Burundi": "Africa",
    "Cabo Verde": "Africa",
    "Cameroon": "Africa",
    "Central African Republic": " Africa",
    "Chad": "Africa",
    "Comoros": "Africa",
    "Congo": "Africa",
    "Côte d'Ivoire": "Africa",
    "Democratic Republic of the Congo": "Africa",
    "Djibouti": "Africa",
    "Egypt": "Africa",
    "Equatorial Guinea": "Africa",
    "Eritrea": "Africa",
    "Eswatini": "Africa",
    "Ethiopia": "Africa",
    "Gabon": "Africa",
    "Gambia": "Africa",
    "Ghana": "Africa",
    "Guinea": "Africa",
    "Guinea-Bissau": "Africa",
    "Kenya": "Africa",
    "Lesotho": "Africa",
    "Liberia": "Africa",
    "Libya": "Africa",
    "Madagascar": "Africa",
    "Malawi": "Africa",
    "Mali": "Africa",
    "Mauritania": "Africa",
    "Mauritius": "Africa",
    "Morocco": "Africa",
    "Mozambique": "Africa",
    "Namibia": "Africa",
    "Niger": "Africa",
    "Nigeria": "Africa",
    "Rwanda": "Africa",
    "Sao Tome and Principe": "Africa",
    "Senegal": "Africa",
    "Seychelles": "Africa",
    "Sierra Leone": "Africa",
    "Somalia": "Africa",
    "South Africa": "Africa",
    "South Sudan": "Africa",
    "Sudan": "Africa",
    "Tanzania": "Africa",
    "Togo": "Africa",
    "Tunisia": "Africa",
    "Uganda": "Africa",
    "Zambia": "Africa",
    "Zimbabwe": "Africa",
    "RSA": "Africa"
}

def assign_group(country: str) -> str:
    return continent_mapping.get(country)

aov['Group'] = aov['Country'].apply(assign_group)
aov['AOV'] = aov['AOV'].round(2)

continent_summary = aov.groupby('Group', as_index=False)['AOV'].mean()
continent_summary = continent_summary.sort_values(by="AOV", ascending=False)

fig_overview = px.bar(
    continent_summary,
    x="Group",
    y="AOV",
    color="Group",
    color_discrete_map={"Asia": "orange", "Europe": "blue"}, 
    title="มูลค่าคำสั่งซื้อโดยเฉลี่ยรายทวีป ( หน่วย : £ )"
)
fig_overview.update_layout(
    xaxis_title="ทวีป",
    yaxis_title="มูลค่าคำสั่งซื้อโดยเฉลี่ย ( หน่วย : £ )"
)
st.plotly_chart(fig_overview, use_container_width=True)

aov_asia = aov[aov['Group'] == "Asia"].sort_values(by="AOV", ascending=False)
aov_europe = aov[aov['Group'] == "Europe"].sort_values(by="AOV", ascending=False)

for df_aov, title, key in [
    (aov_asia, "มูลค่าคำสั่งซื้อโดยเฉลี่ยรายประเทศที่อยู่ในทวีปเอเชีย ( หน่วย : £ )", "asia"),
    (aov_europe, "มูลค่าคำสั่งซื้อโดยเฉลี่ยรายประเทศที่อยู่ในทวีปยุโรป ( หน่วย : £ )", "europe"),
]:
    df_aov['AOV'] = df_aov['AOV'].round(2)
    fig = px.bar(
        df_aov,
        x="Country",
        y="AOV",
        color="Country",
        title=title,
    )
    fig.update_layout(
        xaxis_title="ประเทศ",
        yaxis_title="มูลค่าคำสั่งซื้อโดยเฉลี่ย ( หน่วย : £ )"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- AI Insight: AOV ----
st.subheader("🤖 AI Insights: AOV แบ่งตามทวีป")
mode_aov_ai = st.radio(
    "โหมดการแสดงผล (AOV รายทวีป)",
    ["แสดงกราฟอย่างเดียว", "ให้ AI วิเคราะห์ส่วนนี้"],
    horizontal=True,
    key="mode_aov_ai",
)
if mode_aov_ai == "ให้ AI วิเคราะห์ส่วนนี้":
    with st.spinner("AI กำลังวิเคราะห์ AOV รายทวีป..."):
        client = get_groq_client(groq_api_key)
        prompt = build_aov_group_insight(continent_summary)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "คุณเป็นผู้เชี่ยวชาญด้าน Pricing และ AOV"},
                {"role": "user", "content": prompt},
            ],
        )
        insight = completion.choices[0].message.content
    st.markdown(insight)

st.divider()

# ====================================================
# SECTION 4: KPI + Cancel + Retention
# ====================================================
cancel_query = """
    WITH InvoiceNoC as ( 
        SELECT * 
        FROM df
        WHERE InvoiceNo LIKE 'C%' )

    , InvoiceNoCount as (
        SELECT 
            InvoiceNo,
            SUM(-1*(Quantity * UnitPrice)) AS InvoiceSalesPerInvoiceNo
        FROM InvoiceNoC
        GROUP BY InvoiceNo )
    
    SELECT count(InvoiceNo) as total_cancel_invoices ,
           ROUND(SUM(InvoiceSalesPerInvoiceNo), 2) as sum ,
           ROUND(AVG(InvoiceSalesPerInvoiceNo), 2) AS AOV 
    FROM InvoiceNoCount
""" 

Cancel_all = duckdb.query(cancel_query).to_df()

st.header("💡 Key Insights")

col1, col2, col3 = st.columns(3)
with col1:
    total_purchases = con.execute(
        "SELECT COUNT(DISTINCT InvoiceNo) FROM df_table WHERE Quantity > 0"
    ).fetchone()[0]
    st.metric("คำสั่งซื้อรวม", f"{total_purchases:,} รายการ")
with col2:
    total_customers = con.execute(
        "SELECT COUNT(DISTINCT CustomerID) FROM df_table WHERE CustomerID IS NOT NULL"
    ).fetchone()[0]
    st.metric("จำนวนลูกค้ารวม", f"{total_customers:,} ราย")
with col3:
    total_quantity = con.execute(
        "SELECT SUM(Quantity) FROM df_table WHERE Quantity > 0"
    ).fetchone()[0]
    st.metric("จำนวนสินค้าที่ขายได้", f"{total_quantity:,.0f} ชิ้น")

col4, col5, col6 = st.columns(3)
with col4:
    cancel_count = Cancel_all['total_cancel_invoices'].iloc[0] if len(Cancel_all) > 0 else 0
    st.metric("คำสั่งซื้อที่ยกเลิก", f"{int(cancel_count):,} รายการ")
with col5:
    cancel_sum = Cancel_all['sum'].iloc[0] if len(Cancel_all) > 0 else 0
    st.metric("มูลค่ารวมที่ยกเลิก", f"£{cancel_sum:,.2f}")
with col6:
    cancel_aov = Cancel_all['AOV'].iloc[0] if len(Cancel_all) > 0 else 0
    st.metric("มูลค่าเฉลี่ยต่อคำสั่งซื้อที่ยกเลิก", f"£{cancel_aov:,.2f}")

col7, _, _ = st.columns(3)
with col7:
    cancel_ratio = (
        cancel_count / (total_purchases + cancel_count) * 100
        if total_purchases > 0 else 0
    )
    st.metric("สัดส่วนคำสั่งซื้อที่ยกเลิก", f"{cancel_ratio:.2f}%")

st.header("🔄 Customer Retention Pattern Analysis")
query_retention = """
SELECT 
    CustomerID,
    COUNT(DISTINCT Month) as MonthsActive,
    MIN(Month) as FirstPurchaseMonth,
    MAX(Month) as LastPurchaseMonth
FROM df_table
WHERE Quantity > 0 AND CustomerID IS NOT NULL
GROUP BY CustomerID
HAVING COUNT(DISTINCT Month) >= 2
"""
retention_data = con.execute(query_retention).df()

if len(retention_data) > 0:
    c1, c2 = st.columns(2)
    with c1:
        st.metric("ลูกค้ากลับมาซื้อซ้ำ", f"{len(retention_data):,} ราย")
    with c2:
        avg_months = retention_data['MonthsActive'].mean()
        st.metric("ช่วงเวลาเฉลี่ยที่ลูกค้ากลับมาซื้อซ้ำ", f"{avg_months:.1f} เดือน")

fig_dist = px.histogram(
    retention_data,
    x='MonthsActive',
    title='การแจงแจกแสดงจำนวนเดือนที่ลูกค้ากลับมาซื้อซ้ำ',
    labels={'MonthsActive': 'จำนวนเดือนที่ลูกค้ากลับมาซื้อซ้ำ'}
)
fig_dist.update_layout(height=450, yaxis_title='จำนวนลูกค้า')
st.plotly_chart(fig_dist, use_container_width=True)

# ---- AI Insight: KPI + Retention ----
st.subheader("🤖 AI Insights: KPI, Cancellation และ Retention")
mode_kpi_ai = st.radio(
    "โหมดการแสดงผล (KPI & Retention)",
    ["แสดงตัวเลขอย่างเดียว", "ให้ AI วิเคราะห์ส่วนนี้"],
    horizontal=True,
    key="mode_kpi_ai",
)
if mode_kpi_ai == "ให้ AI วิเคราะห์ส่วนนี้":
    with st.spinner("AI กำลังวิเคราะห์ KPI และ Retention..."):
        client = get_groq_client(groq_api_key)
        prompt = build_kpi_retention_insight(
            total_purchases,
            total_customers,
            total_quantity,
            cancel_count,
            cancel_sum,
            cancel_aov,
            cancel_ratio,
            retention_data,
        )
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "คุณเป็นผู้เชี่ยวชาญด้าน Business Analytics และ CRM"},
                {"role": "user", "content": prompt},
            ],
        )
        insight = completion.choices[0].message.content
    st.markdown(insight)

con.close()

st.divider()

# ====================================================
# SECTION 5: Pareto Analysis
# ====================================================
st.header("🔑 Pareto Analysis ")
st.markdown("Pareto Analysis คือกลุ่มสินค้า 20% แรก ที่สร้างยอดขาย 80% จากยอดขายทั้งหมด")

pareto_query = """
WITH cleaned AS (
    SELECT
        StockCode,
        Description,
        SUM(Quantity) AS TotalQty,                      
        SUM(Quantity * UnitPrice) AS TotalSales         
    FROM df
    WHERE InvoiceNo NOT LIKE 'C%'  
    GROUP BY StockCode, Description
)
SELECT *
FROM cleaned
ORDER BY TotalSales DESC;
"""
stock_sales = duckdb.query(pareto_query).to_df()

stock_sales['CumulativeSales'] = stock_sales['TotalSales'].cumsum()
total_sales = stock_sales['TotalSales'].sum()
stock_sales['CumulativePercent'] = 100 * stock_sales['CumulativeSales'] / total_sales

pareto_cut = stock_sales[stock_sales['CumulativePercent'] <= 80]

c1, c2 = st.columns(2)
with c1:
    product_count = len(pareto_cut)
    total_products = len(stock_sales)
    product_percent = (product_count * 100 / total_products)
    st.metric("จำนวนสินค้า", f"{product_count:,} รายการ")
    st.markdown(f"คิดเป็น {product_percent:.2f}% จากทั้งหมด {total_products:,} รายการ")
with c2:
    total_sales_80 = pareto_cut['TotalSales'].sum()
    cumulative_percent = pareto_cut['CumulativePercent'].max()
    st.metric("ยอดขายรวม", f"£{total_sales_80:,.2f}")
    st.markdown(f"คิดเป็น {cumulative_percent:.2f}% ของยอดขายทั้งหมด")

def categorize(description):
    d = description.lower()
    categories = {
        "ของตกแต่งบ้าน": ["metal", "wood", "frame", "sign", "plaque", "heart", "garland", "wreath", "wall", "hanging", "cushion"],
        "ของใช้ในครัว": ["mug", "cup", "plate", "bowl", "jar", "jug", "tin", "kitchen", "baking", "cake", "teapot", "cutlery"],
        "แฟชั่น": ["mirror", "cosmetic", "purse", "wallet", "keyring", "scarf", "jewellery"],
        "งานฝีมือ": ["craft", "felt", "notebook", "pencil", "pen", "stamp", "colouring", "paper", "card"],
        "ของเล่น": ["toy", "doll", "jigsaw", "game", "puzzle", "child", "kids"],
        "ของปาร์ตี้": ["party", "gift bag", "gift", "wrapping", "ribbon", "balloon", "birthday"],
        "เซ็ตของขวัญ": ["lunch", "box set", "tin set", "food box", "snack box", "storage box"],
        "ของตกแต่งเทศกาล": ["christmas", "easter", "halloween", "advent", "festive", "snow", "santa"],
        "เครื่องหอม": ["candle", "incense", "aroma", "scent"],
        "ของตกแต่งสวน": ["garden", "planter", "flower pot", "watering can"],
        "อุปกรณ์ไฟฟ้า": ["lamp", "light", "lantern", "torch"]
    }
    for category, keywords in categories.items():
        if any(keyword in d for keyword in keywords):
            return category
    return "อื่นๆ"

pareto_cut["Category"] = pareto_cut["Description"].apply(categorize)
duckdb.register("pareto_cut", pareto_cut)

summary = duckdb.query("""
    SELECT
        Category,
        SUM(TotalSales) AS TotalSales,
        SUM(TotalQty) AS ProductCount
    FROM pareto_cut
    GROUP BY Category
""").to_df()

total_sales_pareto = summary["TotalSales"].sum()
total_products_pareto = summary["ProductCount"].sum()

summary["SalesPercent"] = 100 * summary["TotalSales"] / total_sales_pareto
summary["ProductPercent"] = 100 * summary["ProductCount"] / total_products_pareto
summary["is_other"] = (summary["Category"] == "อื่นๆ").astype(int)
summary = summary.sort_values(
    by=["is_other", "SalesPercent"],
    ascending=[True, False]
).drop(columns="is_other").reset_index(drop=True)

summary.index = range(1, len(summary) + 1)

st.subheader("สรุปยอดขายตามหมวดสินค้า")
st.markdown(f"รายการสินค้า {product_percent:.2f}% สามารถจำแนกหมวดสินค้าได้ดังนี้ ")
st.dataframe(
    summary.style.format({
        "TotalSales": "{:.2f}",
        "ProductCount": "{:,.0f}",
        "SalesPercent": "{:.2f}",
        "ProductPercent": "{:.2f}"
    })
)

# ---- AI Insight: Pareto ----
st.subheader("🤖 AI Insights: Pareto และ หมวดสินค้า")
mode_pareto_ai = st.radio(
    "โหมดการแสดงผล (Pareto Analysis)",
    ["แสดงตารางอย่างเดียว", "ให้ AI วิเคราะห์ส่วนนี้"],
    horizontal=True,
    key="mode_pareto_ai",
)
if mode_pareto_ai == "ให้ AI วิเคราะห์ส่วนนี้":
    with st.spinner("AI กำลังวิเคราะห์ Pareto และหมวดสินค้า..."):
        client = get_groq_client(groq_api_key)
        prompt = build_pareto_insight(summary)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "คุณเป็นผู้เชี่ยวชาญด้าน Category Management และ Merchandising"},
                {"role": "user", "content": prompt},
            ],
        )
        insight = completion.choices[0].message.content
    st.markdown(insight)

# Footer
st.divider()
st.caption("Page 2")
