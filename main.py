import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text



def create_streamlit_app(chain, portfolio):
    st.title("📧 Cold Mail Generator")

    url_input = st.text_input(
        "Enter a careers page or job URL:",
        value="https://www.google.com/about/careers/applications/jobs/results"
    )

    submit_button = st.button("Generate Emails")

    if submit_button and url_input:
        try:
            with st.spinner("Scraping and analyzing jobs..."):
                loader = WebBaseLoader([url_input])
                page_data = loader.load().pop().page_content
                cleaned_data = clean_text(page_data)

                portfolio.load_portfolio()
                jobs = chain.extract_jobs(cleaned_data)

            if not jobs:
                st.warning("No jobs found on this page.")
                return

            for idx, job in enumerate(jobs, start=1):
                st.subheader(f"📌 Job {idx}: {job.get('role')}")

                # 🔥 IMPORTANT: derive semantic query text
                query_text = (
                    job.get("short_description")
                    or job.get("description")
                    or job.get("role")
                )

                links = portfolio.query_links(query_text)

                email = chain.write_mail(job, links)

                st.code(email, language="markdown")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    st.set_page_config(
        layout="wide",
        page_title="Cold Email Generator",
        page_icon="📧"
    )

    chain = Chain()
    portfolio = Portfolio()

    create_streamlit_app(chain, portfolio)
