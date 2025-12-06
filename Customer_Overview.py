import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq   # ใช้ Groq สำหรับ AI Insight
from streamlit_gsheets import GSheetsConnection

# ---------------------------------------------------
# Page config
# ---------------------------------------------------
st.set_page_config(page_title="Customer Overview", page_icon="🌍", layout="wide")
st.title("💻 E-commerce Analysis")
st.title("🌍 ภาพรวมลูกค้า (Customer Overview)")

# ---------------------------------------------------
# API Key สำหรับ AI Insight
# ---------------------------------------------------
groq_api_key = "MY_API_KEY"

# ---------------------------------------------------
# AI Helper Functions
# ---------------------------------------------------
@st.cache_resource
def get_groq_client(api_key: str):
    return Groq(api_key=api_key)


def build_country_value_insight_prompt(top10_df: pd.DataFrame, all_df: pd.DataFrame) -> str:
    """
    ใช้สร้าง prompt ให้ AI วิเคราะห์ Top 10 ประเทศตามมูลค่าคำสั่งซื้อรวม
    """
    rows_text = "\n".join(
        f"- {row.country}: มูลค่ารวม £{row.value_by_country:,.0f} | "
        f"ธุรกรรม {row.transaction_count:,} ครั้ง | ปริมาณ {row.total_quantity:,} ชิ้น"
        for row in top10_df.itertuples()
    )

    total_countries = len(all_df)
    total_value = all_df["value_by_country"].sum()
    top10_value = top10_df["value_by_country"].sum()
    top10_share = top10_value / total_value * 100 if total_value > 0 else 0

    prompt = f"""
คุณคือ Data Analyst ด้านอีคอมเมิร์ซ
ช่วยวิเคราะห์ข้อมูล "มูลค่าคำสั่งซื้อรวมตามประเทศ" โดยดูเฉพาะ Top 10 ประเทศแรก

ข้อมูล Top 10 ประเทศ (เรียงจากมากไปน้อย):

{rows_text}

บริบทเพิ่มเติม:
- จำนวนประเทศทั้งหมดในชุดข้อมูล: {total_countries} ประเทศ
- มูลค่าคำสั่งซื้่อรวมทั้งหมด: £{total_value:,.2f}
- Top 10 คิดเป็นประมาณ {top10_share:.1f}% ของมูลค่ารวมทั้งหมด

กรุณาสรุปเป็น bullet point ภาษาไทย (ไม่เกิน 6 ข้อ):
- เปรียบเทียบประเทศที่มีมูลค่ารวมสูงสุดกับอันดับถัด ๆ ไป (มีความเหลื่อมล้ำแค่ไหน)
- มองหากลุ่มประเทศที่ตัวเลขใกล้เคียงกัน (เช่น กลุ่มกลาง / กลุ่มท้าย)
- แจ้งให้เห็น pattern ระหว่างมูลค่ารวม, จำนวนธุรกรรม และปริมาณรวม
- เสนอแนวคิดเชิงธุรกิจ 1–2 ข้อ เช่น ควรโฟกัสประเทศใด, ควรเจาะลูกค้าเดิมหรือหาลูกค้าใหม่

ตอบเป็น bullet point เท่านั้น
"""
    return prompt


def build_aov_insight_prompt(df_aov: pd.DataFrame) -> str:
    """
    ใช้สร้าง prompt ให้ AI วิเคราะห์ Top 15 AOV ตามประเทศ
    """
    rows_text = "\n".join(
        f"- {row.Country}: {row.AOV:,.0f} £"
        for row in df_aov.itertuples()
    )

    prompt = f"""
คุณคือ Data Analyst ผู้เชี่ยวชาญด้านอีคอมเมิร์ซ
ช่วยวิเคราะห์ข้อมูลมูลค่าคำสั่งซื้อโดยเฉลี่ย (Average Order Value: AOV) ต่อประเทศด้านล่างนี้

หน่วยเป็นปอนด์ (£) ต่อ 1 ใบเสร็จ:

{rows_text}

กรุณาสรุปเป็น bullet point ภาษาไทย (ไม่เกิน 6 ข้อ):
- ระบุประเทศที่มี AOV สูงสุด/ต่ำสุด และช่วงห่างประมาณเท่าไร
- มองหากลุ่มประเทศที่ AOV ใกล้เคียงกัน (cluster แบบคร่าว ๆ)
- มองภาพรวมว่า Top 3–5 ประเทศแรกมีความโดดเด่นอย่างไร
- เสนอแนวคิดเชิงธุรกิจ 1–2 ข้อ เช่น ควรโฟกัสประเทศไหนเพื่อเพิ่มรายได้ หรือควรสำรวจ insight อะไรต่อ

ตอบเป็น bullet point เท่านั้น
"""
    return prompt


# ---------------------------------------------------
# Load data
# ---------------------------------------------------
@st.cache_data(ttl=60)
def load_data():
    conn = st.connection("gsheets", type=GSheetsConnection)
    return conn.read()

# ---------------------------------------------------
# Main logic
# ---------------------------------------------------
try:
    df = load_data()
    st.success(
        f"✅ โหลดข้อมูลสำเร็จ: {len(df):,} รายการ "
        "(ข้อมูลจาก: UCI Machine Learning Repository https://doi.org/10.24432/C5BW33)"
    )

    # Preview
    with st.expander("🔍 ดูข้อมูลตัวอย่าง"):
        st.dataframe(df.head(10))
        st.write(f"**Columns:** {', '.join(df.columns.tolist())}")

    # DuckDB connection
    con = duckdb.connect(':memory:')

    con.register('df', df)

    # Column names
    selected_country_col = 'Country'
    selected_quantity_col = 'Quantity'
    selected_price_col = 'UnitPrice'
    selected_date_col = 'InvoiceDate'

    # Required columns check
    required_columns = [selected_country_col, selected_quantity_col, selected_price_col]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if not missing_columns:
        # ---------- Date handling ----------
        if selected_date_col in df.columns:
            try:
                con.execute(f"""
                    CREATE OR REPLACE TABLE df_with_date AS
                    SELECT *,
                           TRY_CAST("{selected_date_col}" AS DATE) as parsed_date,
                           EXTRACT(YEAR FROM TRY_CAST("{selected_date_col}" AS DATE)) as year,
                           EXTRACT(MONTH FROM TRY_CAST("{selected_date_col}" AS DATE)) as month
                    FROM df
                """)
                date_filter = ""
                table_name = "df_with_date"
            except Exception as e:
                st.warning(f"⚠️ ไม่สามารถแปลงวันที่ได้: {str(e)}")
                date_filter = ""
                table_name = "df"
        else:
            date_filter = ""
            table_name = "df"

        # ---------- Aggregate by country ----------
        query = f"""
        SELECT 
            "{selected_country_col}" as country,
            SUM("{selected_quantity_col}" * "{selected_price_col}") as value_by_country,
            COUNT(*) as transaction_count,
            SUM("{selected_quantity_col}") as total_quantity
        FROM {table_name}
        WHERE "{selected_country_col}" IS NOT NULL
          AND "{selected_quantity_col}" IS NOT NULL
          AND "{selected_price_col}" IS NOT NULL
          {date_filter}
        GROUP BY "{selected_country_col}"
        ORDER BY value_by_country DESC
        """
        country_data = con.execute(query).df()

        # Top 10 + others
        top_10 = country_data.head(10).copy()
        others_value = country_data.iloc[10:]['value_by_country'].sum() if len(country_data) > 10 else 0
        others_transactions = country_data.iloc[10:]['transaction_count'].sum() if len(country_data) > 10 else 0
        others_quantity = country_data.iloc[10:]['total_quantity'].sum() if len(country_data) > 10 else 0

        if others_value > 0:
            others_row = pd.DataFrame([{
                'country': 'Others',
                'value_by_country': others_value,
                'transaction_count': others_transactions,
                'total_quantity': others_quantity
            }])
            chart_data = pd.concat([top_10], ignore_index=True)
        else:
            chart_data = top_10

        chart_data.index = range(1, len(chart_data) + 1)

        # ---------- Summary metrics ----------
        st.divider()
        st.subheader("📈 สถิติ")

        col3, col4, col5 = st.columns([1, 1, 2])

        with col3:
            st.metric(
                "ค่าเฉลี่ยมูลค่าคำสั่งซื้อต่อประเทศ",
                f"£{country_data['value_by_country'].mean():,.2f}"
            )
        with col4:
            top10_pct = top_10['value_by_country'].sum() / country_data['value_by_country'].sum() * 100
            st.metric("Top 10 คิดเป็น", f"{top10_pct:.1f}%")
        with col5:
            if others_value > 0:
                others_pct = others_value / country_data['value_by_country'].sum() * 100
                st.metric("ประเทศอื่น ๆ คิดเป็น", f"{others_pct:.1f}%")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.metric("จำนวนประเทศทั้งหมด", len(country_data))
        with col2:
            st.metric("มูลค่าคำสั่งซื้อรวมทั้งหมด", f"£{country_data['value_by_country'].sum():,.2f}")
        with col3:
            st.metric(
                "ประเทศที่มีมูลค่าคำสั่งซื้อมากที่สุด",
                f"{country_data.iloc[0]['country']} (£{country_data.iloc[0]['value_by_country']:,.2f})"
            )
        
        
        st.divider()

        # ---------- Layout: Table + Map ----------
        col1, col2 = st.columns([1, 1])

        # ----- LEFT: Top 10 table + bar -----
        with col1:
            st.subheader("📊 Top 10 ประเทศแบ่งตามมูลค่าคำสั่งซื้อ")

            fig_bar = px.bar(
                chart_data,
                x='country',
                y='value_by_country',
                title='Top 10 ประเทศแบ่งตามมูลค่าคำสั่งซื้อรวม',
                labels={'country': 'ประเทศ', 'value_by_country': 'มูลค่ารวม'},
                color='value_by_country',
                color_continuous_scale='Reds',
                text='value_by_country'
            )
            fig_bar.update_traces(
                texttemplate='%{text:,.0f}',
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>มูลค่ารวม: £%{y:,.2f}<extra></extra>'
            )
            fig_bar.update_layout(
                xaxis_tickangle=-45,
                showlegend=False,
                height=500,
                yaxis_title='มูลค่าคำสั่งซื้อรวม ( หน่วย : £ )'
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            chart_data_display = chart_data.rename(columns={
                'country': 'ประเทศ',
                'value_by_country': 'มูลค่ารวม (£)',
                'transaction_count': 'จำนวนธุรกรรม',
                'total_quantity': 'ปริมาณรวม'
            })
        st.dataframe(
            chart_data_display.style.format({
                'มูลค่ารวม (£)': '{:,.2f}',
                'จำนวนธุรกรรม': '{:,.0f}',
                'ปริมาณรวม': '{:,.0f}'
            }),
            use_container_width=True,
            height=400
        )

        # ----- RIGHT: Map + summary -----
        with col2:
            st.subheader("🗺️ แผนที่โลกแสดงมูลค่าคำสั่งซื้อตามประเทศ")

            fig_map = px.choropleth(
                country_data,
                locations='country',
                locationmode='country names',
                color='value_by_country',
                hover_name='country',
                hover_data={
                    'value_by_country': ':,.2f',
                    'transaction_count': ':,',
                    'total_quantity': ':,'
                },
                color_continuous_scale='bluyl',
                labels={'value_by_country': 'มูลค่ารวม'}
            )
            fig_map.update_layout(
                geo=dict(
                    showframe=True,
                    showcoastlines=True,
                    projection_type='natural earth'
                ),
                height=500,
                margin={"r": 0, "t": 0, "l": 0, "b": 0}
            )
            st.plotly_chart(fig_map, use_container_width=True)


        # ---------- AI Insight: Top 10 Country Value ----------
        st.subheader("🤖 AI Insights: Top 10 ประเทศตามมูลค่าคำสั่งซื้อรวม")

        mode_country = st.radio(
            "โหมดการแสดงผล (มูลค่าคำสั่งซื้อตามประเทศ)",
            ["แสดงข้อมูลอย่างเดียว", "ให้ AI วิเคราะห์ข้อมูลนี้"],
            horizontal=True,
            key="mode_country_insight",
        )

        if mode_country == "ให้ AI วิเคราะห์ข้อมูลนี้":
            with st.spinner("AI กำลังวิเคราะห์ Top 10 ประเทศตามมูลค่าคำสั่งซื้อรวม..."):
                client = get_groq_client(groq_api_key)
                prompt_country = build_country_value_insight_prompt(top_10, country_data)
                completion_country = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    messages=[
                        {
                            "role": "system",
                            "content": "คุณเป็นผู้เชี่ยวชาญด้านการวิเคราะห์ข้อมูลลูกค้าและธุรกิจอีคอมเมิร์ซ",
                        },
                        {
                            "role": "user",
                            "content": prompt_country,
                        },
                    ],
                )
                insight_country = completion_country.choices[0].message.content

            st.markdown(insight_country)

        # ---------------------------------------------------
        # AOV BY COUNTRY (Top 15)
        # ---------------------------------------------------
        st.divider()
        st.subheader("📊 มูลค่าคำสั่งซื้อโดยเฉลี่ยแบ่งตามประเทศ (Average Order Value: AOV)")

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

        aov_all = con.execute(aov_query).df()
        top15_countries = aov_all.sort_values(by="AOV", ascending=False).head(15).copy()
        top15_countries["AOV"] = top15_countries["AOV"].round(2)
        
        fig_bar_aov = px.bar(
            top15_countries,
            x="Country",
            y="AOV",
            color="AOV",
            color_continuous_scale="Blues",
            title="Top 15 ประเทศแบ่งตามมูลค่าคำสั่งซื้อโดยเฉลี่ย ( หน่วย : £ )"
        )
        fig_bar_aov.update_layout(
            xaxis_title="ประเทศ",
            yaxis_title="มูลค่าคำสั่งซื้อโดยเฉลี่ย ( หน่วย : £ )"
        )
        st.plotly_chart(fig_bar_aov, use_container_width=True)

        # ---------- AI Insight: AOV ----------
        st.subheader("🤖 AI Insights: AOV แบ่งตามประเทศ")

        mode_aov = st.radio(
            "โหมดการแสดงผล (AOV ต่อประเทศ)",
            ["แสดงกราฟอย่างเดียว", "ให้ AI วิเคราะห์กราฟนี้"],
            horizontal=True,
            key="mode_aov_insight",
        )

        if mode_aov == "ให้ AI วิเคราะห์กราฟนี้":
            with st.spinner("AI กำลังวิเคราะห์ข้อมูล AOV ตามประเทศ..."):
                client = get_groq_client(groq_api_key)
                prompt_aov = build_aov_insight_prompt(top15_countries)
                completion_aov = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    messages=[
                        {
                            "role": "system",
                            "content": "คุณเป็นผู้เชี่ยวชาญด้านการวิเคราะห์ข้อมูลลูกค้าและธุรกิจอีคอมเมิร์ซ",
                        },
                        {
                            "role": "user",
                            "content": prompt_aov,
                        },
                    ],
                )
                insight_aov = completion_aov.choices[0].message.content

            st.markdown(insight_aov)

    else:
        st.error("❌ ไม่พบ column ที่จำเป็นในข้อมูล")
        st.write("**Columns ที่มี:**", df.columns.tolist())
        st.write("**Columns ที่ขาด:**", missing_columns)
        st.info("💡 กรุณาตรวจสอบว่าข้อมูลมี column: Country, Quantity, และ UnitPrice")

except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
    import traceback
    st.code(traceback.format_exc())

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.divider()
st.caption("Page 1")
