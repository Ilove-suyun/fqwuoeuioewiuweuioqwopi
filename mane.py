import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# 페이지 기본 설정
st.set_page_config(
    page_title="서울 100년 기온 변화 분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 100년 변화 추이")
st.markdown("지난 100여 년간 서울의 연평균 기온 변화를 한눈에 파악할 수 있는 대화형 분석 앱입니다.")

# 데이터 불러오기 함수 (캐싱 처리)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"
    try:
        df = pd.read_csv(url, encoding='utf-8')
    except Exception:
        df = pd.read_csv(url, encoding='cp949')
    
    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()
    
    # 컬럼명 매핑
    col_map = {}
    for col in df.columns:
        if '날짜' in col:
            col_map[col] = '날짜'
        elif '평균기온' in col:
            col_map[col] = '평균기온'
        elif '최저기온' in col:
            col_map[col] = '최저기온'
        elif '최고기온' in col:
            col_map[col] = '최고기온'
        elif '지점' in col:
            col_map[col] = '지점'
    
    df = df.rename(columns=col_map)
    
    # 날짜 데이터 변환
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    df = df.dropna(subset=['날짜'])
    
    # 연도 추출
    df['연도'] = df['날짜'].dt.year
    
    # 수치형 컬럼 변환
    for col in ['평균기온', '최저기온', '최고기온']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

with st.spinner("데이터를 로드 중입니다..."):
    df = load_data()

# 연도별 집계 (연평균 기온)
yearly_df = df.groupby('연도').agg(
    연평균기온=('평균기온', 'mean'),
    평균최저기온=('최저기온', 'mean'),
    평균최고기온=('최고기온', 'mean'),
    연중최저기온=('최저기온', 'min'),
    연중최고기온=('최고기온', 'max'),
    관측일수=('평균기온', 'count')
).reset_index()

# 관측일수가 부족한 연도 제외 (300일 이상 데이터 보장)
yearly_df = yearly_df[yearly_df['관측일수'] >= 300].copy()

# 10년 이동평균 계산
yearly_df['10년이동평균'] = yearly_df['연평균기온'].rolling(window=10, min_periods=3).mean()

# 사이드바 설정
st.sidebar.header("⚙️ 분석 옵션")

min_year = int(yearly_df['연도'].min())
max_year = int(yearly_df['연도'].max())

start_year, end_year = st.sidebar.slider(
    "조회 연도 범위",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

show_ma = st.sidebar.checkbox("10년 이동평균선 표시", value=True)
show_trend = st.sidebar.checkbox("장기 추세선(선형 회귀) 표시", value=True)

# 선택된 기간 데이터 필터링
filtered_df = yearly_df[(yearly_df['연도'] >= start_year) & (yearly_df['연도'] <= end_year)]

# 핵심 요약 지표 카드
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="기간 내 평균 기온", 
        value=f"{filtered_df['연평균기온'].mean():.2f} ℃"
    )

with col2:
    max_row = filtered_df.loc[filtered_df['연평균기온'].idxmax()]
    st.metric(
        label="최고 연평균 기온", 
        value=f"{int(max_row['연도'])}년",
        delta=f"{max_row['연평균기온']:.2f} ℃"
    )

with col3:
    min_row = filtered_df.loc[filtered_df['연평균기온'].idxmin()]
    st.metric(
        label="최저 연평균 기온", 
        value=f"{int(min_row['연도'])}년",
        delta=f"{min_row['연평균기온']:.2f} ℃"
    )

with col4:
    if len(filtered_df) >= 20:
        first_10 = filtered_df.head(10)['연평균기온'].mean()
        last_10 = filtered_df.tail(10)['연평균기온'].mean()
        diff = last_10 - first_10
        st.metric(
            label="초기 대비 기온 변화", 
            value=f"{diff:+.2f} ℃"
        )
    else:
        st.metric(label="분석 연수", value=f"{len(filtered_df)}년")

st.markdown("---")

# 그래프 시각화
st.subheader("📈 연도별 기온 변화 그래프")

fig = go.Figure()

# 연평균 기온 선
fig.add_trace(go.Scatter(
    x=filtered_df['연도'],
    y=filtered_df['연평균기온'],
    mode='lines+markers',
    name='연평균 기온',
    line=dict(color='#E74C3C', width=2),
    marker=dict(size=4),
    hovertemplate='%{x}년: %{y:.2f}℃<extra></extra>'
))

# 10년 이동평균선
if show_ma:
    fig.add_trace(go.Scatter(
        x=filtered_df['연도'],
        y=filtered_df['10년이동평균'],
        mode='lines',
        name='10년 이동평균',
        line=dict(color='#2980B9', width=3, dash='dash'),
        hovertemplate='%{x}년 (10년 평균): %{y:.2f}℃<extra></extra>'
    ))

# 선형 추세선
if show_trend and len(filtered_df) > 1:
    x = filtered_df['연도']
    y = filtered_df['연평균기온']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    fig.add_trace(go.Scatter(
        x=x,
        y=p(x),
        mode='lines',
        name='장기 추세선',
        line=dict(color='#27AE60', width=2, dash='dot'),
        hovertemplate='추세선: %{y:.2f}℃<extra></extra>'
    ))

fig.update_layout(
    title=f"서울 연평균 기온 추이 ({start_year}년 ~ {end_year}년)",
    xaxis_title="연도",
    yaxis_title="기온 (℃)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    height=520
)

st.plotly_chart(fig, use_container_width=True)

# 상세 데이터표
with st.expander("📊 연도별 데이터 상세보기"):
    st.dataframe(
        filtered_df[['연도', '연평균기온', '평균최저기온', '평균최고기온', '연중최저기온', '연중최고기온']]
        .style.format({
            '연도': '{:.0f}',
            '연평균기온': '{:.2f} ℃',
            '평균최저기온': '{:.2f} ℃',
            '평균최고기온': '{:.2f} ℃',
            '연중최저기온': '{:.1f} ℃',
            '연중최고기온': '{:.1f} ℃'
        }),
        use_container_width=True
    )
