from dotenv import load_dotenv
import os

# Load environment variables at the very start
load_dotenv(override=True)

# Verify API key is loaded and print the actual loaded key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No API key found in environment variables")
print(f"Loaded API key starts with: {api_key[:15]}...")

# Set the API key explicitly
os.environ["OPENAI_API_KEY"] = api_key

from crewai import Agent, Task, Crew
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI

# Initialize the language model with the loaded API key
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4",
    temperature=0.7,
)

# Initialize basic tools
search_tool = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Initialize role-specific LLMs
researcher_llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4",
    temperature=0.7,  # Higher temperature for creative research
)

data_engineer_llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4",
    temperature=0.2,  # Lower temperature for precise data processing
)

report_builder_llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4",
    temperature=0.5,  # Balanced temperature for report writing
)

# Create the Research Agent with enhanced backstory
researcher = Agent(
    role='Senior Research Analyst',
    goal='Conduct thorough research and gather relevant data',
    backstory="""You are a senior research analyst with 15 years of experience in data analysis
    and market research. You have a PhD in Data Science and specialize in pattern recognition
    and trend analysis. You excel at:
    1. Finding and validating information from multiple sources
    2. Identifying emerging trends and patterns
    3. Critical evaluation of data quality and relevance
    4. Providing context and insights for complex topics""",
    verbose=True,
    tools=[search_tool, wikipedia],
    allow_delegation=True,
    llm=researcher_llm  # Using role-specific LLM
)

# Create the Data Engineer Agent with enhanced backstory
data_engineer = Agent(
    role='Data Engineer',
    goal='Process and transform data into useful formats',
    backstory="""You are a senior data engineer with expertise in data architecture
    and pipeline development. Your specialties include:
    1. Data cleaning and normalization
    2. Handling missing values and outliers
    3. Data quality assurance
    4. Creating efficient data transformation workflows
    You always ensure data integrity while preparing it for analysis.""",
    verbose=True,
    allow_delegation=True,
    llm=data_engineer_llm  # Using role-specific LLM
)

# Create the Report Builder Agent with enhanced backstory
report_builder = Agent(
    role='Report Builder',
    goal='Create comprehensive reports with visualizations',
    backstory="""You are an expert data visualization specialist and technical writer
    with a keen eye for detail. Your strengths include:
    1. Creating clear and impactful visualizations
    2. Translating complex data into understandable insights
    3. Structuring information for maximum clarity
    4. Maintaining consistency in reporting standards
    You excel at making complex information accessible to all stakeholders.""",
    verbose=True,
    allow_delegation=True,
    llm=report_builder_llm  # Using role-specific LLM
)

# Define tasks for each agent
research_task = Task(
    description="""Research the given topic thoroughly. 
    Gather relevant data and information from reliable sources.
    Provide a summary of key findings.""",
    agent=researcher
)

process_task = Task(
    description="""Clean and process the collected data.
    Identify and handle any missing values or anomalies.
    Prepare the data for analysis and visualization.""",
    agent=data_engineer
)

report_task = Task(
    description="""Create a comprehensive report using the processed data.
    Include relevant visualizations and insights.
    Ensure the report is clear and actionable.""",
    agent=report_builder
)

# Create the crew
data_science_crew = Crew(
    agents=[researcher, data_engineer, report_builder],
    tasks=[research_task, process_task, report_task],
    verbose=2
)

# Function to run the crew
def analyze_topic(topic):
    try:
        # Create a task list with the specific topic
        tasks = [
            Task(
                description=f"""Research {topic} thoroughly. 
                Gather relevant data and information from reliable sources.
                Provide a summary of key findings.""",
                agent=researcher
            ),
            Task(
                description=f"""Clean and process the collected data about {topic}.
                Identify and handle any missing values or anomalies.
                Prepare the data for analysis and visualization.""",
                agent=data_engineer
            ),
            Task(
                description=f"""Create a comprehensive report about {topic} using the processed data.
                Include relevant visualizations and insights.
                Ensure the report is clear and actionable.""",
                agent=report_builder
            )
        ]

        # Create the crew with the updated tasks
        crew = Crew(
            agents=[researcher, data_engineer, report_builder],
            tasks=tasks,
            verbose=2
        )

        # Call kickoff without arguments
        result = crew.kickoff()
        return result
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage
if __name__ == "__main__":
    print("\nVerifying API key:", api_key[:15], "...")
    topic = "Electric Vehicle Market Trends 2024"
    result = analyze_topic(topic)
    print("\nFinal Result:", result)