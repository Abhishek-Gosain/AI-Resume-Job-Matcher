
import re
import html
from io import BytesIO
from xml.sax.saxutils import escape

import fitz
import ollama
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="AI Resume & Job Matcher",
    page_icon="📄",
    layout="wide",
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .skills-box {
        height: 220px;
        overflow-y: auto;
        border: 1px solid #d6d6d6;
        border-radius: 8px;
        padding: 12px;
        font-size: 14px;
        line-height: 1.8;
    }

    .job-description-box {
        max-height: 250px;
        overflow-y: auto;
        border: 1px solid #d6d6d6;
        border-radius: 8px;
        padding: 12px;
        font-size: 14px;
        line-height: 1.7;
        white-space: pre-wrap;
    }

    .small-note {
        font-size: 13px;
        color: #777;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "llama3.2:1b"

SKILLS = [
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "C++",
    "C#",
    "SQL",
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "Machine Learning",
    "Deep Learning",
    "Artificial Intelligence",
    "Generative AI",
    "LLM",
    "LLMs",
    "RAG",
    "Embeddings",
    "Semantic Search",
    "NLP",
    "Prompt Engineering",
    "Vector Database",
    "Vector Databases",
    "LangChain",
    "Streamlit",
    "FastAPI",
    "Flask",
    "REST API",
    "REST APIs",
    "Git",
    "GitHub",
    "Docker",
    "AWS",
    "GCP",
    "Azure",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "Agile",
    "Scrum",
    "Project Management",
    "Communication",
    "Problem Solving",
]


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

DEFAULT_STATE = {
    "analysis": None,
    "feedback": None,
    "ranked_jobs": None,
    "analysis_error": None,
    "feedback_error": None,
    "ranking_error": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# TEXT EXTRACTION
# --------------------------------------------------

def extract_resume_text(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        pdf_bytes = uploaded_file.getvalue()

        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            text = "\n".join(
                page.get_text() for page in document
            )

        return text

    if file_name.endswith(".txt"):
        return uploaded_file.getvalue().decode(
            "utf-8",
            errors="replace",
        )

    raise ValueError("Please upload a PDF or TXT resume.")


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------
# SKILL DETECTION
# --------------------------------------------------

def find_skills(text):
    found_skills = []
    text_lower = text.lower()

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"

        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return sorted(set(found_skills), key=str.lower)


# --------------------------------------------------
# RESUME ANALYSIS
# --------------------------------------------------

def analyze_resume(resume_text, job_description):
    model = load_embedding_model()

    resume_text = clean_text(resume_text)
    job_description = clean_text(job_description)

    if not resume_text:
        raise ValueError(
            "No readable text was found in the resume."
        )

    if not job_description:
        raise ValueError(
            "Please enter a job description."
        )

    embeddings = model.encode(
        [resume_text, job_description],
        convert_to_numpy=True,
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]],
    )[0][0]

    score = round(
        max(0, min(100, float(similarity) * 100)),
        2,
    )

    resume_skills = find_skills(resume_text)
    job_skills = find_skills(job_description)

    matching_skills = [
        skill for skill in job_skills
        if skill in resume_skills
    ]

    missing_skills = [
        skill for skill in job_skills
        if skill not in resume_skills
    ]

    return {
        "score": score,
        "resume_text": resume_text,
        "job_description": job_description,
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
    }


# --------------------------------------------------
# LOCAL AI FEEDBACK USING OLLAMA
# --------------------------------------------------

def generate_ai_feedback(analysis):
    resume_text = analysis["resume_text"][:10000]
    job_description = analysis["job_description"][:6000]

    prompt = f"""
You are a practical resume reviewer and interview coach.

Review the resume against the job description below.

Write in clear, natural English. Keep your suggestions specific
to the information provided. Do not invent experience, skills,
qualifications, achievements, or results.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

SIMILARITY SCORE:
{analysis["score"]}%

MATCHING SKILLS:
{", ".join(analysis["matching_skills"]) or "None detected"}

SKILLS TO REVIEW:
{", ".join(analysis["missing_skills"]) or "None detected"}

Please provide these sections:

## 1. Match Explanation
Explain how well the resume matches the job and mention the
main strengths and gaps.

## 2. Resume Improvement Suggestions
Give practical suggestions to improve the resume for this role.
Suggest wording improvements only when supported by the person's
actual experience. Do not invent facts.

## 3. Interview Questions
Provide five relevant interview questions based on the role
and the resume.

## 4. Preparation Tips
Suggest a short list of topics the candidate should prepare.
"""

    response = ollama.chat(
        model=LLM_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": 0.3,
        },
    )

    return response["message"]["content"]


# --------------------------------------------------
# COMPARE MULTIPLE JOBS
# --------------------------------------------------

def compare_jobs(resume_text, job_descriptions):
    model = load_embedding_model()

    texts = [clean_text(resume_text)] + [
        clean_text(job) for job in job_descriptions
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
    )

    resume_embedding = embeddings[0]
    job_embeddings = embeddings[1:]

    scores = cosine_similarity(
        [resume_embedding],
        job_embeddings,
    )[0]

    results = []

    for index, (job, score) in enumerate(
        zip(job_descriptions, scores),
        start=1,
    ):
        results.append(
            {
                "Job": f"Job {index}",
                "Match Score (%)": round(
                    max(0, min(100, float(score) * 100)),
                    2,
                ),
                "Job Description": job,
            }
        )

    results.sort(
        key=lambda item: item["Match Score (%)"],
        reverse=True,
    )

    return results


# --------------------------------------------------
# PDF REPORT GENERATION
# --------------------------------------------------

def create_analysis_report(analysis, feedback=None):
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontSize=20,
            leading=25,
            alignment=TA_CENTER,
            spaceAfter=16,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontSize=13,
            leading=17,
            spaceBefore=12,
            spaceAfter=7,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            spaceAfter=6,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
            textColor=colors.grey,
            spaceAfter=6,
        )
    )

    story = []

    def add_paragraph(text, style="ReportBody"):
        safe_text = escape(str(text)).replace(
            "\n",
            "<br/>",
        )

        story.append(
            Paragraph(
                safe_text or "Not provided",
                styles[style],
            )
        )

    def add_section(title):
        story.append(
            Paragraph(
                escape(title),
                styles["SectionHeading"],
            )
        )

    story.append(
        Paragraph(
            "AI Resume &amp; Job Matcher",
            styles["ReportTitle"],
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey,
        )
    )

    story.append(Spacer(1, 10))

    add_section("Analysis Summary")

    summary_data = [
        ["Metric", "Result"],
        [
            "Resume Similarity Score",
            f'{analysis["score"]}%',
        ],
        [
            "Matching Skills",
            str(len(analysis["matching_skills"])),
        ],
        [
            "Skills to Review",
            str(len(analysis["missing_skills"])),
        ],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[2.5 * inch, 3.5 * inch],
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#243B53"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.lightgrey,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(summary_table)

    story.append(Spacer(1, 8))

    add_paragraph(
        "Note: This similarity score measures text similarity. "
        "It is not a probability of getting hired and does not "
        "replace a recruiter's assessment.",
        "SmallNote",
    )

    add_section("Matching Skills")

    if analysis["matching_skills"]:
        add_paragraph(
            ", ".join(analysis["matching_skills"])
        )
    else:
        add_paragraph(
            "No matching skills were detected from the skill list."
        )

    add_section("Skills to Review")

    if analysis["missing_skills"]:
        add_paragraph(
            ", ".join(analysis["missing_skills"])
        )
    else:
        add_paragraph(
            "No additional skills were detected from the skill list."
        )

    add_section("Resume Skills Detected")

    add_paragraph(
        ", ".join(analysis["resume_skills"]) or "None detected"
    )

    add_section("Job Description")

    add_paragraph(analysis["job_description"])

    if feedback:
        add_section("AI Suggestions and Interview Preparation")

        # Remove common Markdown formatting for a readable PDF.
        plain_feedback = feedback

        plain_feedback = re.sub(
            r"(?m)^#{1,6}\s*",
            "",
            plain_feedback,
        )

        plain_feedback = plain_feedback.replace("**", "")
        plain_feedback = plain_feedback.replace("__", "")
        plain_feedback = plain_feedback.replace("`", "")

        for line in plain_feedback.splitlines():
            line = line.strip()

            if line.startswith(("- ", "* ")):
                line = "• " + line[2:]

            if line:
                add_paragraph(line)
            else:
                story.append(Spacer(1, 4))

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# --------------------------------------------------
# DOWNLOAD POPUP CALLBACK
# --------------------------------------------------

def show_download_popup():
    st.toast(
        "Your PDF report is ready to download!",
        icon="📄",
    )


# --------------------------------------------------
# APP HEADER
# --------------------------------------------------

st.title("📄 AI Resume & Job Matcher")

st.write(
    "Upload your resume, compare it with a job description, "
    "and get AI-powered suggestions to improve your application."
)

st.caption(
    "Your resume analysis and AI feedback are generated locally "
    "using Sentence Transformers and Ollama."
)


# --------------------------------------------------
# RESUME AND JOB DESCRIPTION INPUTS
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    uploaded_resume = st.file_uploader(
        "Upload your resume",
        type=["pdf", "txt"],
        help="Supported formats: PDF and TXT.",
    )

with col2:
    job_description = st.text_area(
        "Paste the job description",
        height=220,
        placeholder=(
            "Paste the job responsibilities, required skills, "
            "qualifications, and experience here..."
        ),
    )


# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

if st.button(
    "🔍 Analyze Resume",
    use_container_width=True,
):
    st.toast(
        "Resume analysis started!",
        icon="🔍",
    )

    st.session_state.analysis_error = None
    st.session_state.feedback_error = None
    st.session_state.ranking_error = None

    if uploaded_resume is None:
        st.session_state.analysis_error = (
            "Please upload your resume first."
        )

    elif not job_description.strip():
        st.session_state.analysis_error = (
            "Please enter a job description."
        )

    else:
        try:
            with st.spinner(
                "Reading your resume and comparing it with the job..."
            ):
                resume_text = extract_resume_text(
                    uploaded_resume
                )

                analysis = analyze_resume(
                    resume_text,
                    job_description,
                )

                st.session_state.analysis = analysis

                # Clear results based on the previous analysis.
                st.session_state.feedback = None
                st.session_state.ranked_jobs = None
                st.session_state.feedback_error = None
                st.session_state.ranking_error = None

            st.toast(
                "Resume analysis completed!",
                icon="✅",
            )

        except Exception as error:
            st.session_state.analysis_error = str(error)
            st.session_state.analysis = None

if st.session_state.analysis_error:
    st.error(st.session_state.analysis_error)


# --------------------------------------------------
# RESULTS TABS
# --------------------------------------------------

tab_match, tab_ai, tab_compare = st.tabs(
    [
        "📊 Resume Match",
        "🤖 AI Suggestions",
        "⚖️ Compare Jobs",
    ]
)


# --------------------------------------------------
# TAB 1: RESUME MATCH
# --------------------------------------------------

with tab_match:
    analysis = st.session_state.analysis

    if analysis:
        st.subheader("Resume Match Results")

        score_col, matching_col, missing_col = st.columns(3)

        with score_col:
            st.metric(
                "Similarity Score",
                f'{analysis["score"]}%',
            )

        with matching_col:
            st.metric(
                "Matching Skills",
                len(analysis["matching_skills"]),
            )

        with missing_col:
            st.metric(
                "Skills to Review",
                len(analysis["missing_skills"]),
            )

        st.caption(
            "The similarity score measures text similarity, "
            "not your probability of getting hired."
        )

        st.divider()

        skills_col1, skills_col2 = st.columns(2)

        with skills_col1:
            st.subheader("✅ Matching Skills")

            if analysis["matching_skills"]:
                matching_html = "<br>".join(
                    "• " + html.escape(skill)
                    for skill in analysis["matching_skills"]
                )
            else:
                matching_html = (
                    "No matching skills were detected."
                )

            st.markdown(
                f'<div class="skills-box">{matching_html}</div>',
                unsafe_allow_html=True,
            )

        with skills_col2:
            st.subheader("📌 Skills to Review")

            if analysis["missing_skills"]:
                missing_html = "<br>".join(
                    "• " + html.escape(skill)
                    for skill in analysis["missing_skills"]
                )
            else:
                missing_html = (
                    "No additional skills were detected."
                )

            st.markdown(
                f'<div class="skills-box">{missing_html}</div>',
                unsafe_allow_html=True,
            )

        st.subheader("📄 Job Description")

        with st.expander("View job description"):
            safe_job_description = html.escape(
                analysis["job_description"]
            )

            st.markdown(
                (
                    '<div class="job-description-box">'
                    f"{safe_job_description}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

        st.divider()

        st.subheader("Download Your Report")

        st.write(
            "Download a PDF containing your match score, "
            "skills comparison, job description, and AI feedback "
            "if you have generated it."
        )

        try:
            pdf_report = create_analysis_report(
                analysis,
                st.session_state.feedback,
            )

            st.download_button(
                label="📥 Download Analysis Report (PDF)",
                data=pdf_report,
                file_name="resume_analysis_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                on_click=show_download_popup,
            )

        except Exception as error:
            st.error(
                "Unable to create the PDF report. "
                f"Details: {error}"
            )

    else:
        st.info(
            "Upload a resume, enter a job description, "
            "and click Analyze Resume to see your results."
        )


# --------------------------------------------------
# TAB 2: AI SUGGESTIONS
# --------------------------------------------------

with tab_ai:
    st.subheader("AI-Powered Resume Suggestions")

    st.write(
        "Get a match explanation, resume improvement suggestions, "
        "interview questions, and preparation tips."
    )

    if st.session_state.analysis:
        if st.button(
            "✨ Generate AI Suggestions",
            use_container_width=True,
        ):
            st.toast(
                "Generating AI suggestions...",
                icon="🤖",
            )

            st.session_state.feedback_error = None

            try:
                with st.spinner(
                    "Ollama is reviewing your resume..."
                ):
                    feedback = generate_ai_feedback(
                        st.session_state.analysis
                    )

                    st.session_state.feedback = feedback

                st.toast(
                    "AI suggestions are ready!",
                    icon="✅",
                )

            except Exception as error:
                st.session_state.feedback_error = (
                    "Unable to generate AI suggestions. "
                    f"Details: {error}"
                )

        if st.session_state.feedback_error:
            st.error(st.session_state.feedback_error)

        if st.session_state.feedback:
            st.info(
                "✨ AI-generated content — please review "
                "for accuracy before using it."
            )

            st.markdown(
                st.session_state.feedback
            )

    else:
        st.info(
            "Analyze your resume first to generate AI suggestions."
        )


# --------------------------------------------------
# TAB 3: COMPARE MULTIPLE JOBS
# --------------------------------------------------

with tab_compare:
    st.subheader("Compare Your Resume with Multiple Jobs")

    st.write(
        "Paste multiple job descriptions below. Separate each "
        "description with three hyphens (---) on a separate line."
    )

    multiple_jobs = st.text_area(
        "Job descriptions",
        height=250,
        placeholder=(
            "Job 1: Python Developer\n"
            "Responsibilities and required skills...\n"
            "\n"
            "---\n"
            "\n"
            "Job 2: AI Engineer\n"
            "Responsibilities and required skills..."
        ),
        key="multiple_jobs_input",
    )

    if st.button(
        "⚖️ Compare Jobs",
        use_container_width=True,
    ):
        st.toast(
            "Comparing job descriptions...",
            icon="📊",
        )

        st.session_state.ranking_error = None

        if not st.session_state.analysis:
            st.session_state.ranking_error = (
                "Please analyze your resume first."
            )

        elif not multiple_jobs.strip():
            st.session_state.ranking_error = (
                "Please enter at least one job description."
            )

        else:
            try:
                job_descriptions = [
                    job.strip()
                    for job in re.split(
                        r"(?m)^\s*---\s*$",
                        multiple_jobs,
                    )
                    if job.strip()
                ]

                if not job_descriptions:
                    raise ValueError(
                        "No valid job descriptions were found."
                    )

                with st.spinner(
                    "Comparing your resume with each job..."
                ):
                    ranked_jobs = compare_jobs(
                        st.session_state.analysis["resume_text"],
                        job_descriptions,
                    )

                    st.session_state.ranked_jobs = ranked_jobs

                st.toast(
                    "Job comparison completed!",
                    icon="✅",
                )

            except Exception as error:
                st.session_state.ranking_error = str(error)

    if st.session_state.ranking_error:
        st.error(st.session_state.ranking_error)

    if st.session_state.ranked_jobs:
        st.subheader("Job Match Rankings")

        ranking_data = [
            {
                "Rank": index,
                "Job": job["Job"],
                "Match Score (%)": job["Match Score (%)"],
            }
            for index, job in enumerate(
                st.session_state.ranked_jobs,
                start=1,
            )
        ]

        ranking_df = pd.DataFrame(ranking_data)

        st.dataframe(
            ranking_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Higher similarity scores indicate greater text similarity. "
            "Review the full job requirements before applying."
        )

        for index, job in enumerate(
            st.session_state.ranked_jobs,
            start=1,
        ):
            with st.expander(
                f'{index}. {job["Job"]} — '
                f'{job["Match Score (%)"]}% match'
            ):
                st.write(
                    job["Job Description"]
                )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Streamlit, Sentence Transformers, "
    "scikit-learn, Ollama, PyMuPDF, and ReportLab."
)
