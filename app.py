# ==============================================================================
#           ФИНАЛЬНОЕ ПРИЛОЖЕНИЕ v5.0 (ПОЛНЫЙ КОД)
# ==============================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from adjustText import adjust_text
except ImportError:
    st.error("Ошибка: Необходимые библиотеки не установлены. Проверьте ваш файл requirements.txt.")
    st.stop()

# --- 1. НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Бизнес-Аналитик", page_icon="📈", layout="wide")
warnings.filterwarnings('ignore')
#st.set_option('deprecation.showPyplotGlobalUse', False)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- 2. БЛОК АВТОРИЗАЦИИ ---
try:
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("Ошибка: Файл конфигурации 'config.yaml' не найден.")
    st.stop()

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

st.title("👨‍💻 AI Бизнес-Аналитик")

authenticator.login()

if st.session_state["authentication_status"]:
    # ---- ЕСЛИ ПОЛЬЗОВАТЕЛЬ УСПЕШНО ВОШЕЛ ----
    
    with st.sidebar:
        st.write(f'Добро пожаловать, *{st.session_state["name"]}*!')
        authenticator.logout('Выйти', 'main')

    st.header("Загрузите ваш файл для анализа")
    uploaded_file = st.file_uploader("Выберите файл с продажами...", type=['csv', 'xlsx'], label_visibility="collapsed")

    if uploaded_file is not None:
        with st.spinner('Анализирую данные... Это может занять до минуты...'):
            try:
                # --- ЧТЕНИЕ И ПОДГОТОВКА ДАННЫХ ---
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                
                required_columns = ['OrderID', 'OrderDate', 'Dish', 'Price']
                if not all(col in df.columns for col in required_columns):
                    st.error(f"Ошибка: В вашем файле отсутствуют обязательные колонки: {', '.join(required_columns)}")
                    st.stop()
                    
                df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
                st.success(f"✔️ Файл '{uploaded_file.name}' успешно загружен. Найдено {len(df)} строк.")
                
                st.subheader("Предпросмотр данных")
                st.dataframe(df.head())

                # --- ОБЩИЕ KPI ---
                st.header("Ключевые показатели бизнеса 📊")
                total_revenue = df['Price'].sum()
                number_of_orders = df['OrderID'].nunique()
                average_check = total_revenue / number_of_orders if number_of_orders > 0 else 0
                
                kpi_cols = st.columns(3)
                kpi_cols[0].metric("Общая выручка", f"{total_revenue:,.0f} тг".replace(',', ' '))
                kpi_cols[1].metric("Количество заказов", f"{number_of_orders}")
                kpi_cols[2].metric("Средний чек", f"{average_check:,.0f} тг".replace(',', ' '))
                
                # --- АНАЛИЗ ПО ВРЕМЕНИ ---
                st.header("Анализ по времени 🕒")
                daily_sales = df.groupby(df['OrderDate'].dt.date)['Price'].sum()
                st.write("Динамика выручки по дням:")
                st.line_chart(daily_sales)

                # --- МЕНЮ-ИНЖИНИРИНГ ---
                st.header("Матрица Меню-Инжиниринга 🍽️")
                menu_analysis = df.groupby('Dish').agg(Popularity=('Dish', 'count'), Revenue=('Price', 'sum'))
                avg_popularity = menu_analysis['Popularity'].mean()
                avg_revenue = menu_analysis['Revenue'].mean()
                
                fig, ax = plt.subplots(figsize=(14, 10))
                ax.scatter(menu_analysis['Popularity'], menu_analysis['Revenue'], s=120, color='royalblue', alpha=0.6)
                texts = [ax.text(row['Popularity'], row['Revenue'], index, fontsize=10) for index, row in menu_analysis.iterrows()]
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
                ax.axvline(x=avg_popularity, color='r', linestyle='--')
                ax.axhline(y=avg_revenue, color='r', linestyle='--')
                ax.set_title('Матрица Меню-Инжиниринга', fontsize=16)
                ax.set_xlabel('Популярность (Количество продаж)')
                ax.set_ylabel('Выручка (тенге)')
                ax.grid(True)
                st.pyplot(fig)

                # --- АНАЛИЗ "ИДЕАЛЬНЫХ ПАР" ---
                st.header("Анализ 'Идеальных пар' 🧺")
                basket = (df.groupby(['OrderID', 'Dish'])['OrderID'].count().unstack().reset_index().fillna(0).set_index('OrderID'))
                def encode_units(x): return 1 if x >= 1 else 0
                basket_sets = basket.applymap(encode_units)
                
                if basket_sets.shape[1] > 0 and not basket_sets.sum(axis=1).max() < 2:
                    frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
                    if not frequent_itemsets.empty:
                        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
                        if not rules.empty:
                            st.write("Найденные правила 'Если... то...':")
                            st.dataframe(rules.sort_values(by=['lift', 'confidence'], ascending=False)[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
                        else:
                            st.info("Сильных 'связок' между товарами не найдено.")
                    else:
                        st.info("Популярных наборов товаров не найдено.")
                else:
                    st.info("В данных нет чеков с двумя и более товарами для анализа связей.")

            except Exception as e:
                st.error(f"Произошла ошибка при анализе файла. Убедитесь, что формат файла и названия колонок верны. Ошибка: {e}")

elif st.session_state["authentication_status"] == False:
    st.error('Имя пользователя или пароль неверны')
elif st.session_state["authentication_status"] is None:
    st.warning('Пожалуйста, введите имя пользователя и пароль для доступа.')
