# Data Science Crew AI

An AI-powered data science crew that performs market research and data analysis using LangChain and CrewAI.

## Features
- Research automation
- Data analysis and processing
- Report generation with visualizations

## Setup
1. Clone the repository
```bash
git clone https://github.com/fitzroypet/data-scientist-agent.git
cd data-scientist-agent
```

2. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create .env file and add your API key
```bash
OPENAI_API_KEY=your_api_key_here
```

5. Run the application
```bash
python data_science_crew.py
```

## Project Structure
```
data-scientist-agent/
├── data_science_crew.py
├── requirements.txt
├── .env
└── templates/
    └── index.html
```

## License
[MIT License](LICENSE) 