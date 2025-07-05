# views/about_this.py
import streamlit as st


st.title("About this App")

intro = """
This app is designed to help private-equity firms and investors conduct **deep research** on brands and industries.

- **Brand Research** – Analyse specific brands, their market position, and growth potential.  
- **Industry Research** – Explore broader industry trends, competitive landscapes, and investment opportunities.

Use the navigation menu on the left to switch between sections.
"""

st.markdown(intro)
