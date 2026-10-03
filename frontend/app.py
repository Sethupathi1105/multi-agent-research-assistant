import os
import time

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Multi-Agent Research Assistant", layout="wide")
st.title("Multi-Agent Research Assistant")

question = st.text_input("Research question")

if st.button("Run", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
        st.stop()

    try:
        r = requests.post(f"{API_URL}/research", json={"question": question}, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Could not reach the API: {e}")
        st.stop()

    job_id = r.json()["job_id"]

    with st.status("Agents are researching...", expanded=True) as status:
        while True:
            j = requests.get(f"{API_URL}/research/{job_id}", timeout=30).json()
            if j["status"] == "done":
                status.update(label=f"Done in {j['seconds']}s", state="complete")
                break
            if j["status"] == "failed":
                status.update(label="Failed", state="error")
                st.error(j["error"])
                st.stop()
            time.sleep(3)

    st.markdown(j["report"])
    if j.get("claims"):
        with st.expander("Verified claims"):
            st.json(j["claims"])