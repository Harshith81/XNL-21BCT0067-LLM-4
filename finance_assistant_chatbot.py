import os
from typing import List, Dict, Any
from datetime import datetime
import base64
import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import necessary libraries
from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
import phi.model.openai as openai_module
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.file import FileTools
from phi.assistant import Assistant
from phi.document import Document
from phi.knowledge import Knowledge
from phi.tools import Tool
import phi.api
from phi.workplace import Workplace         
from phi.playground import Playground  

os.makedirs("knowledge", exist_ok=True)
os.makedirs("storage", exist_ok=True)
os.makedirs("assistant_storage", exist_ok=True)
os.makedirs("alerts", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# Get environment variables
phi_api_key = os.getenv("PHI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
alphavantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")  

# Set phi API key first
if phi_api_key:
    phi.api.key = phi_api_key

# Validate API keys 
missing_keys = []
if not phi_api_key:
    missing_keys.append("PHI_API_KEY")
if not openai_api_key:
    missing_keys.append("OPENAI_API_KEY")
if not groq_api_key:
    missing_keys.append("GROQ_API_KEY")

if missing_keys:
    print(f"Warning: The following API keys are missing: {', '.join(missing_keys)}")
    print("Some functionality may not work correctly.")


class ImageAnalysisTool(Tool):
    def __init__(self):
        super().__init__(
            name="image_analysis",
            description="Analyzes charts, graphs, and financial visualizations",
            function=self.analyze_image,
        )
    
    def analyze_image(self, image_path: str) -> str:
        """
        Analyze an image using OpenAI's vision capabilities
        
        Args:
            image_path: Path to the image file
            
        Returns:
            A string containing the analysis of the image
        """
        try:
         
            if not openai_api_key:
                return "Error: OpenAI API key is missing. Cannot analyze image."
                
       
            if not os.path.exists(image_path):
                return f"Error: Image file not found at path: {image_path}"
                
           
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            
        
            client = openai_module.OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Analyze this financial chart or graph in detail."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "What does this financial visualization show? Extract key insights, trends, and data points."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

# Custom Data Visualization Tool
class DataVisualizationTool(Tool):
    def __init__(self):
        super().__init__(
            name="data_visualization",
            description="Creates financial visualizations based on data",
            function=self.create_visualization,
        )
    
    def create_visualization(self, data: Dict[str, Any], chart_type: str = "line", title: str = "Financial Data") -> str:
        """
        Create a visualization based on financial data
        
        Args:
            data: Dictionary with financial data
            chart_type: Type of chart to create (line, bar, pie, etc.)
            title: Title for the chart
            
        Returns:
            Path to the saved visualization
        """
        try:
            plt.figure(figsize=(10, 6))
            
            if chart_type == "line":
                for key, values in data.items():
                    if isinstance(values, list) and all(isinstance(x, (int, float)) for x in values):
                        plt.plot(values, label=key)
            elif chart_type == "bar":
                plt.bar(list(data.keys()), list(data.values()))
            elif chart_type == "pie":
                plt.pie(list(data.values()), labels=list(data.keys()), autopct='%1.1f%%')
            
            plt.title(title)
            if chart_type != "pie": 
                plt.legend()
            
            # Save visualization
            output_dir = "visualizations"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
            plt.savefig(output_path)
            plt.close()
            
            return output_path
        except Exception as e:
            return f"Error creating visualization: {str(e)}"

# Custom Sentiment Analysis Tool
class SentimentAnalysisTool(Tool):
    def __init__(self):
        super().__init__(
            name="sentiment_analysis",
            description="Analyzes sentiment of financial news and reports",
            function=self.analyze_sentiment,
        )
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of financial text
        
        Args:
            text: Financial news or report text
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
          
            if not openai_api_key:
                return {"error": "OpenAI API key is missing. Cannot analyze sentiment."}
                
            client = openai_module.OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analyzer. Analyze the following text and provide a sentiment score between -1.0 (extremely negative) and 1.0 (extremely positive), along with key positive and negative points."},
                    {"role": "user", "content": text}
                ]
            )
            
            analysis = response.choices[0].message.content
            
        
            sentiment_score = 0.0
            positive_points = []
            negative_points = []
            
            lower_analysis = analysis.lower()
            if "sentiment score:" in lower_analysis:
                try:
                    score_text = lower_analysis.split("sentiment score:")[1].split("\n")[0]
                    sentiment_score = float(score_text.strip())
                except:
                    pass
            
            if "positive points:" in lower_analysis:
                positive_section = lower_analysis.split("positive points:")[1]
                if "negative points:" in positive_section:
                    positive_section = positive_section.split("negative points:")[0]
                positive_points = [point.strip("- ").strip() for point in positive_section.split('\n') if point.strip().startswith('-')]
            
            if "negative points:" in lower_analysis:
                negative_section = lower_analysis.split("negative points:")[1]
                negative_points = [point.strip("- ").strip() for point in negative_section.split('\n') if point.strip().startswith('-')]
            
            return {
                "sentiment_score": sentiment_score,
                "positive_points": positive_points,
                "negative_points": negative_points,
                "full_analysis": analysis
            }
        except Exception as e:
            return {"error": f"Error analyzing sentiment: {str(e)}"}

# Custom Market Alert Tool
class MarketAlertTool(Tool):
    def __init__(self):
        super().__init__(
            name="market_alert",
            description="Sets up and triggers market alerts based on specified conditions",
            function=self.create_alert,
        )
    
    def create_alert(self, symbol: str, condition: str, threshold: float) -> str:
        """
        Create a market alert based on specified conditions
        
        Args:
            symbol: Stock symbol to monitor
            condition: Condition type (above, below, percent_change)
            threshold: Threshold value for the alert
            
        Returns:
            Confirmation message for the alert
        """
        try:
      
            os.makedirs("alerts", exist_ok=True)
            alert_id = f"alert_{symbol}_{condition}_{threshold}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            alert_file = os.path.join("alerts", "alerts.txt")
            with open(alert_file, "a") as f:
                f.write(f"{alert_id},{symbol},{condition},{threshold},{datetime.now().isoformat()}\n")
            return f"Alert created successfully. Alert ID: {alert_id}"
        except Exception as e:
            return f"Error creating alert: {str(e)}"


def create_sample_knowledge_docs():
    """Create sample knowledge base documents if they don't exist."""
    os.makedirs("knowledge", exist_ok=True)
    
    with open("knowledge/investment_basics.md", "w") as f:
        f.write("""# Investment Basics
        
## Types of Investments
- Stocks: Ownership shares in public companies
- Bonds: Debt securities where you lend money to entities
- ETFs: Baskets of securities traded like stocks
- Mutual Funds: Professionally managed investment portfolios
- Real Estate: Physical property investments

## Risk and Return
Risk and expected return are directly related. Higher potential returns typically come with higher risk.

## Diversification
Spreading investments across various asset classes to reduce risk.
        """)
    
    # Technical analysis document
    with open("knowledge/technical_analysis.md", "w") as f:
        f.write("""# Technical Analysis
        
## Common Indicators
- Moving Averages: Show average price over specific time periods
- RSI (Relative Strength Index): Measures momentum
- MACD (Moving Average Convergence Divergence): Trend-following momentum indicator
- Bollinger Bands: Volatility indicator

## Chart Patterns
- Head and Shoulders: Reversal pattern
- Double Tops/Bottoms: Reversal patterns
- Triangles: Continuation patterns
- Flags: Short-term consolidation patterns
        """)
    
    # Fundamental analysis document
    with open("knowledge/fundamental_analysis.md", "w") as f:
        f.write("""# Fundamental Analysis
        
## Key Financial Metrics
- Earnings Per Share (EPS)
- Price-to-Earnings Ratio (P/E)
- Revenue Growth
- Debt-to-Equity Ratio

## Company Valuation
Assessing a company's intrinsic value based on its financial statements, market conditions, and growth potential.
        """)
        
    # Tax strategies document
    with open("knowledge/tax_strategies.md", "w") as f:
        f.write("""# Tax Strategies
        
## Common Tax Strategies
- Tax-Loss Harvesting
- Retirement Account Contributions
- Capital Gains Optimization

## Considerations
Understand the local tax regulations and consult a tax professional for personalized advice.
        """)
    
    # Retirement planning document
    with open("knowledge/retirement_planning.md", "w") as f:
        f.write("""# Retirement Planning
        
## Key Considerations
- Savings Goals
- Investment Risk Tolerance
- Diversification of Retirement Accounts
- Inflation Impact

## Strategies
- 401(k) and IRA Contributions
- Annuities and Pension Plans
- Diversified Investment Portfolios
        """)


create_sample_knowledge_docs()

agents_available = True

if not groq_api_key:
    print("Warning: Groq API key missing. Web Search Agent and Finance Agent will not function.")
    agents_available = False

if not openai_api_key:
    print("Warning: OpenAI API key missing. Multi-Modal Agent and Advisor Agent will not function.")
    agents_available = False

if agents_available:
    # Web Search Agent (for financial news and sentiment analysis)
    web_search_agent = Agent(
        name="Web Search Agent",
        role="Financial information researcher and news analyzer",
        model=Groq(api_key=groq_api_key, model="llama3-70b-8192"),
        tools=[
            DuckDuckGo(),
            SentimentAnalysisTool(),
            FileTools()
        ],
        instructions=[
            "Search the web for the latest financial information and news.",
            "Analyze sentiment of financial news and reports.",
            "Always include sources and publication dates.",
            "Focus on reliable financial sources like Bloomberg, Financial Times, Wall Street Journal, etc.",
            "Provide balanced perspectives with both bull and bear cases.",
            "Indicate the reliability and timeliness of information."
        ],
        show_tool_calls=True,
        markdown=True,
    )

    # Financial Data Agent (for retrieving and visualizing financial data)
    finance_agent = Agent(
        name="Financial Data Agent",
        role="Financial data analyst and visualization expert",
        model=Groq(api_key=groq_api_key, model="llama3-70b-8192"),
        tools=[
            YFinanceTools(
                stock_price=True, 
                analyst_recommendations=True, 
                stock_fundamentals=True,
                company_news=True,
                historical_data=True,
                balance_sheet=True,
                cash_flow=True,
                income_statement=True,
                major_holders=True,
                institutional_holders=True
            ),
            DataVisualizationTool(),
            MarketAlertTool(),
            FileTools()
        ],
        instructions=[
            "Use tables to display financial data clearly.",
            "Create visualizations for complex financial data.",
            "Provide insightful analysis of financial metrics.",
            "Compare financial data to industry benchmarks.",
            "Highlight key trends and anomalies in data.",
            "Calculate relevant financial ratios and explain their significance."
        ],
        show_tool_calls=True,
        markdown=True,
    )

    # Multi-Modal Analysis Agent (for image/chart analysis)
    multimodal_agent = Agent(
        name="Multi-Modal Analysis Agent",
        role="Financial chart and visualization interpreter",
        model=OpenAIChat(api_key=openai_api_key, model="gpt-4o"),
        tools=[
            ImageAnalysisTool(),
            DataVisualizationTool(),
            FileTools()
        ],
        instructions=[
            "Analyze financial charts, graphs, and visualizations.",
            "Identify key patterns, trends, and anomalies in visuals.",
            "Extract numerical data from charts when possible.",
            "Explain technical indicators present in financial charts.",
            "Relate visual data to relevant financial concepts."
        ],
        show_tool_calls=True,
        markdown=True,
    )

    # Financial Advisor Agent (for personalized financial advice)
    advisor_agent = Agent(
        name="Financial Advisor Agent",
        role="Personal financial advisor and strategist",
        model=OpenAIChat(api_key=openai_api_key, model="gpt-4o"),
        tools=[FileTools()],
        instructions=[
            "Provide personalized financial advice based on user goals and risk profile.",
            "Explain financial concepts in clear, accessible language.",
            "Consider tax implications, time horizons, and diversification.",
            "Always include disclaimers about financial advice limitations.",
            "Ask clarifying questions when user goals are unclear.",
            "Present multiple options when appropriate.",
            "Explain pros and cons of different financial strategies."
        ],
        show_tool_calls=True,
        markdown=True,
    )

    # Master Financial Assistant (coordinates all agents)
    financial_assistant = Agent(
        name="Financial Intelligence Assistant",
        role="Comprehensive financial intelligence and advisory system",
        model=OpenAIChat(api_key=openai_api_key, model="gpt-4o"),
        team=[web_search_agent, finance_agent, multimodal_agent, advisor_agent],
        instructions=[
            "Coordinate specialized agents to provide comprehensive financial intelligence.",
            "Integrate real-time market data with personalized financial advice.",
            "Always include sources for information and data.",
            "Use tables and visualizations to make complex data understandable.",
            "Present balanced perspectives on financial topics.",
            "Adapt to the user's level of financial literacy.",
            "Maintain confidentiality of user financial information.",
            "Clearly distinguish between facts, opinions, and advice.",
            "Flag high-risk financial suggestions with appropriate warnings."
        ],
        show_tool_calls=True,
        markdown=True,
    )

    financial_knowledge = Knowledge(
        documents=[
            Document(uri="knowledge/investment_basics.md"),
            Document(uri="knowledge/technical_analysis.md"),
            Document(uri="knowledge/fundamental_analysis.md"),
            Document(uri="knowledge/tax_strategies.md"),
            Document(uri="knowledge/retirement_planning.md")
        ],
        storage_dir="./storage"
    )


    enhanced_assistant = Assistant(
        name="Financial Intelligence Suite",
        agent=financial_assistant,
        knowledge=financial_knowledge,
        storage=AssistantStorage(persist_dir="./assistant_storage"),
        description="Comprehensive financial intelligence system with real-time data analysis, visualization, and personalized advice"
    )

    
    financial_workplace = Workplace(
        name="Financial Intelligence Workplace",
        assistants=[enhanced_assistant],
        description="Complete financial intelligence ecosystem for real-time analysis and advice"
    )

   
    financial_playground = Playground(
        agents=[web_search_agent, finance_agent, multimodal_agent, advisor_agent, financial_assistant],
        assistants=[enhanced_assistant]
    ).get_app()
    
else:
    print("Warning: Required API keys are missing. Creating placeholder objects.")

    financial_playground = None
    enhanced_assistant = None
    financial_workplace = None


if __name__ == "__main__":
    print("Setting up Financial Assistant components...")
    
    create_sample_knowledge_docs()
    print("Sample knowledge base documents created.")     
    
    if agents_available:
        print("\nFinancial Assistant setup complete! You can now:")
        print("1. Run the playground app for interactive testing")
        print("2. Use the enhanced_assistant directly in your code")  
        print("3. Access individual agents for specific financial tasks")
        
    
        print("\nExample code to run the playground app:")
        print("    import uvicorn")
        print("    uvicorn.run(financial_playground, host='0.0.0.0', port=8000)")
        
        print("\nExample code to use the assistant directly:")
        print("    response = enhanced_assistant.run('What are the current market trends for tech stocks?')")
        print("    print(response)")
    else:
        print("\nFinancial Assistant setup incomplete due to missing API keys.")
        print("Please add the required API keys to your .env file:")
        if not phi_api_key:
            print("- PHI_API_KEY")
        if not openai_api_key:
            print("- OPENAI_API_KEY")
        if not groq_api_key:
            print("- GROQ_API_KEY")
    
    print("\nFinancial Assistant setup process completed!")     
