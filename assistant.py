import os

from dotenv import load_dotenv

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.schema import Document
from llama_index.core.base.llms.types import MessageRole
from llama_index.llms.anthropic import Anthropic

from llama_parse import LlamaParse

load_dotenv()

LLAMA_CLOUD_API_KEY = os.environ.get('LLAMAPARSE_API_KEY')
CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY')

class Assistant:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock_name = ticker.lower().replace(".", "_")

        self.llm = Anthropic(model="claude-3-haiku-20240307", api_key=CLAUDE_API_KEY)
        self.parser = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="markdown")

        self.file_extractor = {".pdf": self.parser}

        Settings.llm = self.llm
        Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
        
        evolution_tool = FunctionTool.from_defaults(fn=self.evolution)
        evolution_perc_tool = FunctionTool.from_defaults(fn=self.evolution_perc)
        cagr_tool = FunctionTool.from_defaults(fn=self.cagr)
        pe_tool = FunctionTool.from_defaults(fn=self.price_earning_ratio)
        self.tools = {"evolution_tool": evolution_tool, 
                      "evolution_perc_tool": evolution_perc_tool, 
                      "cagr_tool": cagr_tool, 
                      "pe_tool": pe_tool}
        self.query_engines = {}

    def create_query_engine_tool_from_document(self, file_path: str, tool_name: str, tool_description: str):
        dir_reader = SimpleDirectoryReader(input_files=[file_path], file_extractor=self.file_extractor)
        documents = dir_reader.load_data(show_progress=True)

        if documents:
            # Index the documents in a vectorStore
            vector_index = VectorStoreIndex.from_documents(documents)
            query_engine_ = vector_index.as_query_engine(response_mode="compact")

            query = "Summarize all information included in the documents? write in bullet points."
            self.information_included = query_engine_.query(query).response

            #Create the Query Engine Tool over the RAG pipeline
            query_engine_tool_ = QueryEngineTool(
                query_engine=query_engine_,
                metadata=ToolMetadata(
                    name=tool_name,
                    description=tool_description,
                ),
            )

            self.tools[tool_name] = query_engine_tool_
            self.query_engines[tool_name] = query_engine_

            return True
        else:
            return False
    
    def create_query_engine_tool_from_md(self, md: str, tool_name: str, tool_description: str):
        doc_ = Document.from_dict({'text': md})
        index_ = VectorStoreIndex.from_documents([doc_])
        
        query_engine_ = index_.as_query_engine(response_mode="compact")

        query_engine_tool_ = QueryEngineTool(
            query_engine=query_engine_,
            metadata=ToolMetadata(
                name=tool_name,
                description=tool_description,
            ),
        )
            
        self.tools[tool_name] = query_engine_tool_
        self.query_engines[tool_name] = query_engine_

        self.agent = None
    
    def create_agent(self):
        system_prompt = f"""You are a usefull asstistant for financial analysis. Answer the question related to that company. 
            Use the {self.stock_name+"_upgrades_downgrades_md"} tool to get upgrades and downgrades of the stocks.
            Carefully read the information and provide a clear and concise answer to the user.
            When you share numbers, make sure to include the units (e.g., millions/billions) and currency.
            Do not use phrases like 'based on my knowledge' or 'depending on the information'.
            do not answer any question that is not related to the {self.ticker} other than greetings. You are also provided with the following tools:
            - evolution tool: to get the evolution from value b to value a. use number for input value a and b.
            - evolution perc tool: to get the evolution in percentage from value b to value a. use number for input value a and b.
            - cagr tool: to get the compound annual growth rate CAGR from value b to value a over n period. use number for input value a and b and n.
            - price_earning_ratio tool: to get the price per earning ratio is a ratio of the share price devided by the earning per share(EPS) value. use number for input price and eps."""
        if self.stock_name+"_company_info" in self.tools:
            system_prompt += f"""\n- {self.stock_name+"_company_info"} tool: to get current information of the stock or company like company name, current stock price, etc."""
        if self.stock_name+"_company_news" in self.tools:
            system_prompt += f"""\n- {self.stock_name+"_company_news"} tool: to get news about the company, you can ask summary of the news using this tools."""
        if self.stock_name+"_analyst_recommendations" in self.tools:
            system_prompt += f"""\n- {self.stock_name+"_analyst_recommendations"} tool: to get analyst recommendations, you can ask insight from analyst recomandations about the stocks."""
        if self.stock_name+"_upgrades_downgrades_md" in self.tools:
            system_prompt += f"""\n- {self.stock_name+"_upgrades_downgrades_md"} tool: to get upgrades and downgrades of the stocks."""
        if self.stock_name+"_annualreport" in self.tools:
            system_prompt += f"""\n- {self.stock_name+"_annualreport"} tool: to get information about annual report. this is the information included in the annual report:\n{self.information_included}"""
            
        agent_worker = FunctionCallingAgentWorker.from_tools(
                list(self.tools.values()), verbose=True, llm=self.llm,
                system_prompt=system_prompt
            )

        self.agent = agent_worker.as_agent()

    def get_chat_history(self):
        return [{"role": chat.role.value, "content": chat.content} for chat in self.agent.chat_history if chat.role in [MessageRole.USER, MessageRole.ASSISTANT] and len(chat.additional_kwargs.get('tool_calls', [])) == 0]
    
    def evolution(self, a: float, b: float) -> float:
        """Evolution from value b to value a"""
        return f"{a-b}"

    def evolution_perc(self, a: float, b: float) -> float:
        """Evolution in percentage from value b to value a"""
        return f"{round(100*(a/b-1),0)}%"

    def cagr(self, a: float, b: float, n: int) -> float:
        """Compound annual growth rate CAGR from value b to value a over n period"""
        return f"{round(100*((a/b)**(1/n)-1),0)}%"

    def price_earning_ratio(self, price: float, eps: float) -> float:
        """Price per earning ratio is a ratio of the share price devided by the earning per share(EPS) value"""
        return price/eps