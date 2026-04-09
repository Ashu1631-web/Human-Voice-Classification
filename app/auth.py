import streamlit as st

USERS = {"admin": "123", "user": "123"}

def login():
    st.sidebar.title("Login")
    u = st.sidebar.text_input("User")
    p = st.sidebar.text_input("Pass", type="password")

    if st.sidebar.button("Login"):
        if USERS.get(u) == p:
            st.session_state["logged_in"] = True

def logout():
    if st.sidebar.button("Logout"):
        st.session_state.clear()
