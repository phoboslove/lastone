# ==============================================================================
#           ФИНАЛЬНОЕ ПРИЛОЖЕНИЕ v7.0 (ПОЛНЫЙ КОД. ВСЕ ВКЛЮЧЕНО.)
# ==============================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Попытка импортировать опциональные библиотеки
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from adjustText import adjust_text
except ImportError:
    st.error("Ошибка: Необходимые библиотеки (mlxtend, adjustText) не установлены. Пожалуйста, проверьте ваш файл requirements.txt.")
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


# --- 2. СУПЕР-ПРОСТАЯ И НАДЕЖНАЯ СИСТЕМА ПАРОЛЕЙ ---
def check_password():
    """Возвращает `True`, если пользователь ввел правильный пароль."""

    def password_entered():
        """Проверяет, является ли введенный пароль правильным."""
        # Пароль должен быть сохранен в "секретах" Streamlit
        if "password" in st.secrets and st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"] 
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Введите пароль для доступа:", type="password", on_change=password_entered, key="password"
    )
    
    if "password_correct" in st.session_state and not st.session_state["password_correct"] and st.session_state["password"] != '':
        st.error("😕 Пароль неверный.")
    
    return st.session_state.get("password_correct", False)

# --- 3. ОСНОВНАЯ ЛОГИКА ПРИЛОЖЕНИЯ ---
st.title("👨‍💻 AI Бизнес-Аналитик")

if check_password():
    # Этот код выполнится только после ввода правильного пароля
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
                    
                df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce').dt.date
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
                daily_sales = df.groupby(df['OrderDate'])['Price'].sum()
                st.write("Динамика выручки по дням:")
                st.line_chart(daily_sales)

                # --- МЕНЮ-ИНЖИНИРИНГ ---
                st.header("Матрица Меню-Инжиниринга 🍽️")
                menu_analysis = df.groupby('Dish').agg(Popularity=('Dish', 'count'), Revenue=('Price', 'sum'))
                avg_popularity = menu_analysis['Popularity'].mean()
                avg_revenue = menu_analysis['Revenue'].mean()
                
                fig_menu, ax_menu = plt.subplots(figsize=(14, 10))
                ax_menu.scatter(menu_analysis['Popularity'], menu_analysis['Revenue'], s=120, color='royalblue', alpha=0.6)
                texts = [ax_menu.text(row['Popularity'], row['Revenue'], index, fontsize=10) for index, row in menu_analysis.iterrows()]
                adjust_text(texts, ax=ax_menu, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
                ax_menu.axvline(x=avg_popularity, color='r', linestyle='--')
                ax_menu.axhline(y=avg_revenue, color='r', linestyle='--')
                ax_menu.set_title('Матрица Меню-Инжиниринга', fontsize=16)
                ax_menu.set_xlabel('Популярность (Количество продаж)')
                ax_menu.set_ylabel('Выручка (тенге)')
                ax_menu.grid(True)
                st.pyplot(fig_menu)

                # --- АНАЛИЗ "ИДЕАЛЬНЫХ ПАР" ---
                st.header("Анализ 'Идеальных пар' 🧺")
                basket = (df.groupby(['OrderID', 'Dish'])['OrderID'].count().unstack().reset_index().fillna(0).set_index('OrderID'))
                def encode_units(x): return 1 if x >= 1 else 0
                basket_sets = basket.apply(lambda col: col.map(encode_units))
                
                if basket_sets.shape[1] > 0 and not basket_sets.sum(axis=1).max() < 2:
                    frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
                    if not frequent_itemsets.empty:
                        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
                        if not rules.empty:
                            st.write("Найденные правила 'Если... то...':")
                            st.dataframe(rules.sort_values(by=['lift', 'confidence'], ascending=False)[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

            except Exception as e:
                st.error(f"Произошла ошибка при анализе файла. Ошибка: {e}")
