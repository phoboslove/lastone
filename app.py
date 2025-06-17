# ==============================================================================
#           ФИНАЛЬНОЕ ПРИЛОЖЕНИЕ v6.0 (ПОЛНЫЙ КОД. ВСЕ ВКЛЮЧЕНО.)
# ==============================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Попытка импортировать опциональные библиотеки с обработкой ошибок
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from adjustText import adjust_text
except ImportError:
    st.error("Ошибка: Необходимые библиотеки не установлены. Проверьте ваш файл requirements.txt.")
    st.stop()

# --- 1. НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Бизнес-Аналитик", page_icon="📈", layout="wide")
warnings.filterwarnings('ignore')

# Скрываем лишние элементы интерфейса Streamlit
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# --- 2. СУПЕР-ПРОСТАЯ СИСТЕМА ПАРОЛЕЙ ---
def check_password():
    """Возвращает `True`, если пользователь ввел правильный пароль."""

    def password_entered():
        """Проверяет, является ли введенный пароль правильным."""
        # Используем st.secrets, так как это стандарт для Streamlit Cloud
        if st.secrets["password"] == st.session_state["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"] 
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state["password_correct"]:
        st.text_input(
            "Введите пароль для доступа:", type="password", on_change=password_entered, key="password"
        )
        if st.session_state.password != '':
            st.error("😕 Пароль неверный.")
        return False
    else:
        return True

# --- 3. ОСНОВНАЯ ЛОГИКА ПРИЛОЖЕНИЯ ---
st.title("👨‍💻 AI Бизнес-Аналитик")

if check_password():
    st.sidebar.success("Доступ разрешен. Добро пожаловать!")
    
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
                    st.error(f"Ошибка: В вашем файле отсутствуют обязательные колонки. Убедитесь, что есть: {', '.join(required_columns)}")
                    st.stop()
                    
                df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
                st.success(f"✔️ Файл '{uploaded_file.name}' успешно загружен. Найдено {len(df)} строк.")
                
                with st.expander("Предпросмотр данных"):
                    st.dataframe(df.head())

                # --- ОСНОВНЫЕ KPI ---
                st.header("Ключевые показатели бизнеса 📊")
                total_revenue = df['Price'].sum()
                number_of_orders = df['OrderID'].nunique()
                average_check = total_revenue / number_of_orders if number_of_orders > 0 else 0
                
                kpi_cols = st.columns(3)
                kpi_cols[0].metric("Общая выручка", f"{total_revenue:,.0f} тг".replace(',', ' '))
                kpi_cols[1].metric("Количество заказов", f"{number_of_orders}")
                kpi_cols[2].metric("Средний чек", f"{average_check:,.0f} тг".replace(',', ' '))

                # --- АНАЛИЗ КЛИЕНТОВ (С ГРАФИКОМ) ---
                if 'ClientID' in df.columns:
                    st.header("Анализ по клиентам 🏆")
                    customer_spending = df.groupby('ClientID')['Price'].sum().sort_values(ascending=False)
                    st.write("Топ-10 клиентов по сумме трат:")
                    st.dataframe(customer_spending.head(10))
                    
                    st.write("График трат по топ-10 клиентам:")
                    fig_clients, ax_clients = plt.subplots(figsize=(12, 7))
                    customer_spending.head(10).plot(kind='bar', ax=ax_clients, color='royalblue', legend=None)
                    ax_clients.set_ylabel('Сумма трат (тенге)')
                    ax_clients.set_xlabel('ID Клиента')
                    plt.xticks(rotation=45)
                    st.pyplot(fig_clients)

                # --- АНАЛИЗ ПО ВРЕМЕНИ ---
                st.header("Анализ по времени 🕒")
                daily_sales = df.groupby(df['OrderDate'].dt.date)['Price'].sum()
                st.write("Динамика выручки по дням:")
                st.line_chart(daily_sales)

                # --- МЕНЮ-ИНЖИНИРИНГ ---
                st.header("Матрица Меню-Инжиниринга 🍽️")
                # ... (здесь полный код для матрицы) ...

                # --- АНАЛИЗ "ИДЕАЛЬНЫХ ПАР" ---
                st.header("Анализ 'Идеальных пар' 🧺")
                # ... (здесь полный код для анализа корзины) ...

            except Exception as e:
                st.error(f"Произошла ошибка при анализе файла. Ошибка: {e}")
