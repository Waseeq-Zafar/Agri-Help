from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.python import PythonTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
from datetime import datetime
from pydantic import BaseModel
import matplotlib
matplotlib.use('Agg')

load_dotenv()

class AgriculturalChartAgentConfig(BaseModel):
    extra_message: str
    code: str
    image_path: str
    imagekit_url: str = ""  # Optional ImageKit URL

class AgriculturalChartAgent:
    def __init__(self):
        # Remove unsupported kwargs: show_tool_calls, markdown, response_model
        self.agent = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            tools=[PythonTools(), GoogleSearchTools(), YFinanceTools()],
            instructions="""
You are an agricultural data visualization expert. Only use REAL DATA for charts from YFinance and GoogleSearch tools.
Provide extra_message insights, Python code to generate chart, and local image_path.
Always use matplotlib backend 'Agg' and plt.close(). Do NOT use show().
"""
        )

    def generate_response(self, query):
        enhanced_prompt = f"""
Agricultural Query: "{query}"

Instructions:
- Use REAL DATA ONLY (YFinance, GoogleSearch)
- Generate Python code to save chart locally
- Provide insights in extra_message
- Do not use plt.show(), always use matplotlib.use('Agg') and plt.close()
"""
        try:
            print(f"\nProcessing query: {query}")
            response = self.agent.run(enhanced_prompt)

            # Response content handling
            response_content = response.content if hasattr(response, 'content') else response

            # Print insights if available
            if hasattr(response_content, 'extra_message'):
                print(f"\nREAL DATA INSIGHTS:\n{response_content.extra_message}\n{'='*60}")

            # Execute code if available
            if hasattr(response_content, 'code') and response_content.code:
                code_to_execute = response_content.code
                # Remove ``` if present
                if code_to_execute.startswith("```"):
                    code_to_execute = "\n".join(code_to_execute.splitlines()[1:-1])

                exec_globals = {'__builtins__': __builtins__, '__name__': '__main__'}
                exec_locals = {}
                try:
                    exec(code_to_execute, exec_globals, exec_locals)
                    chart_path = exec_locals.get('chart_path') or exec_globals.get('chart_path')
                    if chart_path:
                        print(f"Chart saved at: {chart_path}")
                except Exception as exec_error:
                    print(f"Error executing generated code: {exec_error}")
        except Exception as e:
            print(f"Error in generate_response: {e}")

        return response_content if 'response_content' in locals() else None


def main():
    agent = AgriculturalChartAgent()
    sample_queries = [
        "Compare crop yields over the last decade",
        "Show seasonal rainfall patterns", 
        "Visualize corn vs wheat price trends"
    ]

    while True:
        print("\nSample queries:")
        for i, q in enumerate(sample_queries, 1):
            print(f"{i}. {q}")

        user_query = input("\nEnter your query or 'quit' to exit: ").strip()
        if user_query.lower() in ['quit', 'exit']:
            break
        if user_query.isdigit() and 1 <= int(user_query) <= len(sample_queries):
            user_query = sample_queries[int(user_query) - 1]

        if user_query:
            agent.generate_response(user_query)


if __name__ == "__main__":
    main()
