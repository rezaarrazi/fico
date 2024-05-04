import streamlit as st

import os
from typing import List
import datetime

import yfinance as yf
from duckduckgo_search import DDGS

from llama_index.llms.groq import Groq
from llama_index.core.base.llms.types import MessageRole, ChatMessage

from phi.utils.log import logger

from assistant import Assistant

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

st.set_page_config(
    page_title="FICO",
)
st.title("FICO (Financial Co-pilot)")

def clear_cache():
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)
        
def restart_assistant():
    logger.debug("---*--- Restarting Assistant ---*---")
    st.session_state["research_assistant"] = None
    clear_cache()
    st.rerun()

def main():

    # Get ticker for report
    ticker_to_research = st.sidebar.text_input(
        ":female-scientist: Enter a ticker to research",
        value="BBCA.JK",
    )
    
    # Checkboxes for research options
    st.sidebar.markdown("## Research Options")
    get_company_info = st.sidebar.checkbox("Company Info", value=True)
    get_company_news = st.sidebar.checkbox("Company News", value=True)
    get_analyst_recommendations = st.sidebar.checkbox("Analyst Recommendations", value=True)
    get_upgrades_downgrades = st.sidebar.checkbox("Upgrades/Downgrades", value=True)

    annual_report_tools = st.sidebar.checkbox("Annual Report", value=False)

    # Ticker object
    ticker = yf.Ticker(ticker_to_research)
    stock_name = ticker_to_research.lower().replace(".", "_")

    if annual_report_tools:
        # Add PDFs to knowledge base
        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = 100

        uploaded_file = st.sidebar.file_uploader(
            "Add a PDF :page_facing_up:", type="pdf", key=st.session_state["file_uploader_key"]
        )
    
    # Get the assistant
    research_assistant: Assistant
    if "research_assistant" not in st.session_state or st.session_state["research_assistant"] is None:
        research_assistant = Assistant(ticker=ticker_to_research)
        st.session_state["research_assistant"] = research_assistant
    else:
        research_assistant = st.session_state["research_assistant"]

    # -*- Generate Research Report
    generate_report = st.sidebar.button("Generate Report")

    if "report_generated" not in st.session_state:
        st.session_state["report_generated"] = False

    if generate_report or st.session_state["report_generated"]:
        report_input = ""
        if "report_input" not in st.session_state:
            st.session_state.report_input = ""
        else:
            report_input = st.session_state.report_input
        with st.status("Generating Reports", expanded=True) as status:
            with st.container():
                if report_input == "":
                    if annual_report_tools:
                        if uploaded_file is not None:
                            alert = st.sidebar.info("Processing PDF...", icon="🧠")
                            rag_name = uploaded_file.name.split(".")[0]
                            path_pdf = os.path.join('data', uploaded_file.name)
                            with open(path_pdf, 'wb') as f: 
                                f.write(uploaded_file.getbuffer())

                            if f"{rag_name}_uploaded" not in st.session_state or not st.session_state[f"{rag_name}_uploaded"]:
                                description=f"""Provides Annual Report about {stock_name}
                                    Use a detailed plain text question as input to the tool."""
                                
                                if not research_assistant.create_query_engine_tool_from_document(path_pdf, stock_name+"_annualreport", description):
                                    st.sidebar.error("Could not read PDF")
                                    st.session_state[f"{rag_name}_uploaded"] = True
                                else:
                                    st.session_state[f"{rag_name}_uploaded"] = False

                                os.remove(path_pdf)
                            alert.empty()
                        else:
                            st.error("Please upload the Annual Report PDF or uncheck the box to disable this feature.")
                            st.stop()

                    if get_company_info:
                        company_info_full = ticker.info
                        company_info_md = "## Company Info\n\n"
                        if company_info_full:
                            company_info_cleaned = {
                                "Name": company_info_full.get("shortName"),
                                "Symbol": company_info_full.get("symbol"),
                                "Current Stock Price": f"{company_info_full.get('regularMarketPrice', company_info_full.get('currentPrice'))} {company_info_full.get('currency', 'USD')}",
                                "Market Cap": f"{company_info_full.get('marketCap', company_info_full.get('enterpriseValue'))} {company_info_full.get('currency', 'USD')}",
                                "Sector": company_info_full.get("sector"),
                                "Industry": company_info_full.get("industry"),
                                "Address": company_info_full.get("address1"),
                                "City": company_info_full.get("city"),
                                "State": company_info_full.get("state"),
                                "Zip": company_info_full.get("zip"),
                                "Country": company_info_full.get("country"),
                                "EPS": company_info_full.get("trailingEps"),
                                "P/E Ratio": company_info_full.get("trailingPE"),
                                "52 Week Low": company_info_full.get("fiftyTwoWeekLow"),
                                "52 Week High": company_info_full.get("fiftyTwoWeekHigh"),
                                "50 Day Average": company_info_full.get("fiftyDayAverage"),
                                "200 Day Average": company_info_full.get("twoHundredDayAverage"),
                                "Website": company_info_full.get("website"),
                                "Summary": company_info_full.get("longBusinessSummary"),
                                "Analyst Recommendation": company_info_full.get("recommendationKey"),
                                "Number Of Analyst Opinions": company_info_full.get("numberOfAnalystOpinions"),
                                "Employees": company_info_full.get("fullTimeEmployees"),
                                "Total Cash": company_info_full.get("totalCash"),
                                "Free Cash flow": company_info_full.get("freeCashflow"),
                                "Operating Cash flow": company_info_full.get("operatingCashflow"),
                                "EBITDA": company_info_full.get("ebitda"),
                                "Revenue Growth": company_info_full.get("revenueGrowth"),
                                "Gross Margins": company_info_full.get("grossMargins"),
                                "Ebitda Margins": company_info_full.get("ebitdaMargins"),
                            }
                            for key, value in company_info_cleaned.items():
                                if value:
                                    company_info_md += f"  - {key}: {value}\n\n"
                            report_input += "This section contains information about the company.\n\n"
                        
                            description = f"""Provides current company information about {stock_name}.
                                Use a detailed plain text question as input to the tool."""
                            research_assistant.create_query_engine_tool_from_md(company_info_md, stock_name+"_company_info", description)
                        else:
                            company_info_md += "No information found for this company.\n"
                        
                        report_input += company_info_md
                        report_input += "---\n"
                    if get_company_news:
                        ddgs = DDGS()
                        company_news = ddgs.news(keywords=ticker_to_research+" stocks", max_results=5)
                        company_news_md = "## Company News\n\n\n"
                        if len(company_news) > 0:
                            for news_item in company_news:
                                company_news_md += f"#### {news_item['title']}\n\n"
                                if "date" in news_item:
                                    company_news_md += f"  - Date: {news_item['date']}\n\n"
                                if "url" in news_item:
                                    company_news_md += f"  - Link: {news_item['url']}\n\n"
                                if "source" in news_item:
                                    company_news_md += f"  - Source: {news_item['source']}\n\n"
                                if "body" in news_item:
                                    company_news_md += f"{news_item['body']}"
                                company_news_md += "\n\n"
                                report_input += "This section contains the most recent news articles about the company.\n\n"
                        
                                description = f"""Provides 5 current news about {stock_name}.
                                    No input is required."""
                                research_assistant.create_query_engine_tool_from_md(company_news_md, stock_name+"_company_news", description)
                        else:
                            company_news_md += "No news found for this company.\n"
                                
                        report_input += company_news_md
                        report_input += "---\n"
                            
                    if get_analyst_recommendations:
                        analyst_recommendations = ticker.recommendations
                        report_input += "## Analyst Recommendations\n\n"
                        if not analyst_recommendations.empty:
                            analyst_recommendations_md = analyst_recommendations.to_markdown()
                            report_input += "This table outlines the most recent analyst recommendations for the stock.\n\n"
                            report_input += f"{analyst_recommendations_md}\n"

                            description = f"""Provides analyst recommendation about {stock_name}.
                                No input is required."""
                            research_assistant.create_query_engine_tool_from_md(analyst_recommendations_md, stock_name+"_analyst_recommendations", description)
                        else:
                            report_input += "No analyst recommendations found for this stock.\n"
                        report_input += "---\n"
                        
                    if get_upgrades_downgrades:
                        upgrades_downgrades = ticker.upgrades_downgrades[0:20]
                        report_input += "## Upgrades/Downgrades\n\n"
                        if not upgrades_downgrades.empty:
                            upgrades_downgrades_md = upgrades_downgrades.to_markdown()
                            report_input += "This table outlines the most recent upgrades and downgrades for the stock.\n\n"
                            report_input += f"{upgrades_downgrades_md}\n"

                            description = f"""Provides upgrades and downgrades of {stock_name}.
                                No input is required."""
                            research_assistant.create_query_engine_tool_from_md(upgrades_downgrades_md, stock_name+"_upgrades_downgrades_md", description)
                        else:
                            report_input += "No upgrades or downgrades found for this stock.\n"
                        report_input += "---\n"
                    
                    st.session_state.report_input = report_input

                final_report = ""
                if "final_report" not in st.session_state:
                    st.session_state.final_report = ""
                else:
                    final_report = st.session_state.final_report

                final_report_container = st.empty()
                
                if final_report == "":
                    report_format = """
                        <report_format>
                        ## [Company Name]: Investment Report

                        ### **Overview**
                        {give a brief introduction of the company and why the user should read this report}
                        {make this section engaging and create a hook for the reader}

                        ### Core Metrics
                        {provide a summary of core metrics and show the latest data}
                        - Current price: {current price}
                        - 52-week high: {52-week high}
                        - 52-week low: {52-week low}
                        - Market Cap: {Market Cap} in billions
                        - P/E Ratio: {P/E Ratio}
                        - Earnings per Share: {EPS}
                        - 50-day average: {50-day average}
                        - 200-day average: {200-day average}
                        - Analyst Recommendations: {buy, hold, sell} (number of analysts)

                        ### Financial Performance
                        {provide a detailed analysis of the company's financial performance}

                        ### Growth Prospects
                        {analyze the company's growth prospects and future potential}

                        ### News and Updates
                        {summarize relevant news that can impact the stock price}

                        ### Upgrades and Downgrades
                        {share 2 upgrades or downgrades including the firm, and what they upgraded/downgraded to}
                        {this should be a paragraph not a table}

                        ### [Summary]
                        {give a summary of the report and what are the key takeaways}

                        ### [Recommendation]
                        {provide a recommendation on the stock along with a thorough reasoning}

                        Report generated on: {Month Date, Year (hh:mm AM/PM)}
                        </report_format>
                    """
                    
                    system_prompt = f"""You are a Senior Investment Analyst for Goldman Sachs tasked with producing a research report for a very important client.
                    You will be provided with a stock and information from junior researchers.
                    Carefully read the research and generate a final - Goldman Sachs worthy investment report.
                    Make your report engaging, informative, and well-structured.
                    When you share numbers, make sure to include the units (e.g., millions/billions) and currency.
                    REMEMBER: This report is for a very important client, so the quality of the report is important.
                    Make sure your recommendations are well-supported and backed by data.
                    Make sure your report is properly formatted and follows the <report_format> provided below.
                    If you don't have enough information for a section, you can leave it blank.
                    {report_format}
                    Do not include Goldman Sachs in your report.
                    Current datetime is: {datetime.datetime.now().strftime('%B %d, %Y, %I:%M %p')}
                    """

                    input_message = [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                                    ChatMessage(role=MessageRole.SYSTEM, content=report_input),
                                    ChatMessage(role=MessageRole.USER, content=f"Please generate a report about: {ticker_to_research}\n\n\n")]
                    
                    llm_groq = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
                    final_report = llm_groq.chat(input_message).message.content
                    st.session_state.final_report = final_report

                st.session_state["report_generated"] = True
                final_report_container.markdown(final_report)

                if "agent_created" not in st.session_state or not st.session_state["agent_created"]:
                    research_assistant.create_agent()
                    st.session_state["agent_created"] = True

            status.update(label="Generating Reports", state="complete", expanded=True)

        # Load existing messages
        assistant_chat_history = research_assistant.get_chat_history()
        if len(assistant_chat_history) > 0:
            logger.debug("Loading chat history")
            st.session_state["messages"] = assistant_chat_history
        else:
            logger.debug("No chat history found")
            st.session_state["messages"] = [{"role": "assistant", "content": f"Ask me questions about {ticker_to_research}..."}]
        
        # Prompt for user input
        if prompt := st.chat_input():
            st.session_state["messages"].append({"role": "user", "content": prompt})

        # Display existing chat messages
        for message in st.session_state["messages"]:
            if message["role"] == "system":
                continue
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # If last message is from a user, generate a new response
        last_message = st.session_state["messages"][-1]
        if last_message.get("role") == "user":
            question = last_message["content"]
            with st.chat_message("assistant"):
                response = ""
                resp_container = st.empty()
                response = research_assistant.agent.chat(question)
                response = str(response).split(":")[-1].strip()
                resp_container.markdown(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})

    st.sidebar.markdown("---")
    if st.sidebar.button("New Run"):
        restart_assistant()

if __name__ == "__main__":
    main()