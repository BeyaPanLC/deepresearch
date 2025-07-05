# st_app.py  (entry-point)
import streamlit as st
st.set_page_config(page_title="Deep Research Demo")

# ---- 1. register pages with file-paths ---------------------------
pages = {
    "Deep Research": [
        st.Page(
            "views/research_brand.py",     # <— path to your brand page
            title="Brand Research",
            icon=":material/search:",
            default=True                   # only ONE page should be default
        ),
        st.Page(
            "views/research_industry.py",      # <— industry-wide research
            title="Industry Research",
            icon=":material/insights:"
        ),
    ],
    "Other": [
        st.Page(
            "views/about_this.py",         # <— about page
            title="About this app",
            icon="ℹ️"
        ),
    ],
}

# ---- 2. build the menu and run the chosen page -------------------
pg = st.navigation(pages)
pg.run()                                  # ← crucial!

# ---- 3. global sidebar / footer ---------------------------------
st.sidebar.markdown("## Deep Research App")
st.sidebar.divider()
