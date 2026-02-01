import os
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

load_dotenv()


class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )

    # ---------------------------------------------------------
    # JOB EXTRACTION
    # ---------------------------------------------------------
    def extract_jobs(self, cleaned_text: str) -> List[Dict[str, Any]]:
        """
        Extract job postings from a careers page text.
        Works for Google, Nike, and similar sites.
        """

        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}

            ### INSTRUCTION:
            The text is from a careers page.
            Extract all job postings and return a JSON array.
            Each job object should contain:
            - role
            - company (if available)
            - location (if available)
            - experience_level (if available)
            - short_description

            Only return valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke({"page_data": cleaned_text})

        try:
            parser = JsonOutputParser()
            parsed = parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Unable to parse job listings.")

        return parsed if isinstance(parsed, list) else [parsed]

    # ---------------------------------------------------------
    # INTERNAL: FORMAT LINKS SAFELY
    # ---------------------------------------------------------
    def _format_links(self, links: List[Any]) -> str:
        """
        Handles Chroma outputs:
        - list of dicts
        - list of lists
        - list of strings
        """
        formatted = []

        for item in links:
            if isinstance(item, dict):
                formatted.append(item.get("links"))
            elif isinstance(item, list) and item:
                formatted.append(item[0])
            elif isinstance(item, str):
                formatted.append(item)

        return "\n".join(f"- {l}" for l in formatted if l)

    # ---------------------------------------------------------
    # EMAIL GENERATION
    # ---------------------------------------------------------
    def write_mail(self, job: Dict[str, Any], links: List[Any]) -> str:
        """
        Generate a cold email for a single job.
        """

        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Mohan, a Business Development Executive at AtliQ.

            AtliQ is an AI & Software Consulting company helping enterprises with:
            - AI-driven automation
            - Scalable software systems
            - Process optimization
            - Cost reduction

            Write a concise, professional cold email explaining how
            AtliQ can help fulfill the needs of the role above.

            Include ONLY the most relevant portfolio links below:
            {link_list}

            Constraints:
            - No preamble
            - No subject line
            - Professional business tone

            ### EMAIL:
            """
        )

        job_description = f"""
Role: {job.get('role')}
Company: {job.get('company', 'N/A')}
Location: {job.get('location', 'N/A')}
Experience Level: {job.get('experience_level', 'N/A')}
Description: {job.get('short_description', '')}
"""

        link_list = self._format_links(links)

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({
            "job_description": job_description,
            "link_list": link_list
        })

        return res.content


# ---------------------------------------------------------
# LOCAL TEST (OPTIONAL)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("GROQ KEY LOADED:", bool(os.getenv("GROQ_API_KEY")))
