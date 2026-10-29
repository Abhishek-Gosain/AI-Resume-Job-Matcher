
# AI Resume & Job Matcher

An AI-powered application that helps job seekers compare their resumes with job descriptions, identify matching and missing skills, and get suggestions to improve their resumes.

The application uses Python, semantic similarity, and a locally running LLM to provide resume analysis and interview preparation suggestions.

## Features

- **Resume Upload:** Upload a resume in PDF or TXT format.
- **Resume Matching:** Compare your resume with a job description and calculate a similarity score.
- **Skill Analysis:** Identify matching skills and skills mentioned in the job description that may need attention.
- **AI Suggestions:** Generate a match explanation, resume improvement suggestions, interview questions, and preparation tips.
- **Job Comparison:** Compare your resume against multiple job descriptions and rank them by similarity score.
- **PDF Report:** Download a report containing your analysis, skills comparison, job description, and AI feedback.
- **Popup Notifications:** Receive notifications when performing actions such as resume analysis, AI feedback generation, job comparison, and report download.
- **Local AI Processing:** Use Ollama to generate AI feedback locally.

## Technologies Used

- Python
- Streamlit
- Sentence Transformers
- Scikit-learn
- Ollama
- Llama 3.2 (1B)
- PyMuPDF
- Pandas
- ReportLab

## How It Works

1. Upload your resume in PDF or TXT format.
2. Paste the job description you want to target.
3. Click **Analyze Resume** to calculate the similarity score and compare skills.
4. Open **AI Suggestions** to generate feedback and interview preparation questions.
5. Open **Compare Jobs** to compare multiple job descriptions.
6. Download the analysis report as a PDF.

## Project Structure

```text
AI-Resume-Job-Matcher/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/                 # Local virtual environment; do not upload
```

## Requirements

- Python 3.10 or a compatible version supported by the installed dependencies
- pip
- Ollama
- Git (optional, for version control)

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Abhishek-Gosain/AI-Resume-Job-Matcher.git
cd AI-Resume-Job-Matcher
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install and start Ollama

Download Ollama from:

https://ollama.com/

Pull the required language model:

```bash
ollama pull llama3.2:1b
```

Make sure Ollama is running before generating AI suggestions. Depending on your installation, you may need to start the Ollama application separately.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL shown in your terminal, usually:

http://localhost:8501

## Example Use Cases

- Check how closely a resume matches a job description.
- Identify skills mentioned in a job posting that are not detected in a resume.
- Get suggestions for improving resume content.
- Prepare for interviews based on a target role.
- Compare multiple job opportunities before applying.

## Important Notes

- The similarity score represents textual similarity between the resume and job description. It is not a hiring probability or a guarantee of suitability.
- Skill detection depends on the application's predefined skill list. It may not detect every relevant skill or equivalent term.
- AI-generated suggestions may contain errors and should be reviewed before use.
- Ollama runs the language model locally, but the embedding model may need to be downloaded the first time the application runs.
- Resume and job description content should be reviewed for accuracy and privacy before sharing or using generated reports.

## Future Improvements

- Support for additional resume formats.
- A larger skill database with synonyms and related skills.
- Resume section extraction and analysis.
- Improved job matching with experience and education criteria.
- Resume improvement recommendations tailored to specific roles.
- Optional deployment with secure file handling.

## Author

**Abhishek Gosain**

GitHub: https://github.com/Abhishek-Gosain

---

If you find this project useful, feel free to star the repository.
