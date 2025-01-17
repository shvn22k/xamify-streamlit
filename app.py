import streamlit as st
import asyncio
import os
from typing import Dict
from pdfminer.high_level import extract_text
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.file import FileTools
from typing import Dict


def extract_text_from_pdf(file):
    try:
        from PyPDF2 import PdfReader
        pdf_reader = PdfReader(file)
        text = ''.join(page.extract_text() for page in pdf_reader.pages)
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None


async def save_file(filename, content, result):
    try:
        if content in result:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result[content])
    except Exception as e:
        st.error(f"Error saving file: {e}")


def create_exam_analysis_workflow(api_key: str, syllabus_text: str, question_papers: Dict[int, str]):
    os.environ["GROQ_API_KEY"] = api_key
    groq_model = Groq(id="llama-3.1-70b-versatile")

    #agent 1
    syllabus_agent = Agent(
        name="syllabus parser",
        role="parse the given string to determine the subject and syllabus.",
        model=groq_model,
        instructions=["Always respond in a well-structured format including all 5 units and their detailed topics."]
    )
    #agent 2
    questionpaper_agent = Agent(
        name="question paper parser",
        role="parse the given dictionary to detect questions from each section",
        model=groq_model,
        instructions=[
            "Return a proper dictionary consisting of all the questions mentioned in the given question paper dictionary with year as key and questions string as value.",
            "Return the questions only, exclude any other details."
        ],
    )
    #agent 3
    exam_analyser_agent = Agent(
        name="ai exam analyser",
        role="Analyze syllabus and question papers.",
        model=groq_model,
        instructions=[
            "Use syllabus of subject and analyse question papers to determine the most important and frequently asked topics from each unit.",
            "Always respond in a structured format- Unit Name: important topics.",
            "Always include the number of time the topic has appeared in the question papers."
        ]
    )
    #agent 4
    question_generator_agent = Agent(
        name="question generator",
        role="Generate probable exam questions.",
        model=groq_model,
        instructions=[
            "Generate 10-15 probable questions for upcoming exam based on analysis of past papers and important topics",
            "Include both short answer (2 marks) and long answer (10 marks) questions",
            "Focus on frequently tested topics and important concepts",
            "Format the output as a proper model question paper in markdown",
            "Ensure questions are unique and not directly copied from past papers",
            "Refer to the type and quality of questions from previous year question papers provided",
            "Mark the weightage (2/10 marks) clearly for each question"
        ],
    )

    async def run_workflow():
        try:
            syllabus_result = syllabus_agent.run(f"Parse and structure this syllabus content: {syllabus_text}")
            questions_result = questionpaper_agent.run(f"Parse and structure these question papers: {str(question_papers)}")
            analysis_result = exam_analyser_agent.run(
                f"""Analyse the following:
                Syllabus structure: {syllabus_result.content}\n
                Question paper history: {questions_result.content}
                
                Provide analysis of important topics and their frequency."""
            )
            practice_questions = question_generator_agent.run(
                f"""Based on the following information:
                Question paper patterns: {questions_result.content}
                Topic frequency analysis: {analysis_result.content}
                
                Generate a model question paper with 10-15 questions following these guidelines:
                1. Include both short (5 marks) and long answer (10 marks) questions
                2. Focus on topics that appear frequently in past papers
                3. Cover important topics from all units
                4. Include theoretical and practical questions
                5. Format as a proper question paper with sections and marks
                """
            )
            return {
                "exam_analysis": analysis_result.content,
                "practice_questions": practice_questions.content,
            }
        except Exception as e:
            st.error(f"Damn bro that's a lot of tokens you gave in inputs for a free tier model, maybe reduce the number of question papers.")
            return {f"Error: {e}"}

    return run_workflow

# Streamlit App
async def main():
    st.title("Xamify : Your AI exam aid")

    st.sidebar.header("Upload Files and Keys")
    api_key = st.sidebar.text_input("Enter your Groq API key.", type="password")
    if not api_key:
        st.warning("Please provide a valid Groq API key 🙏🏻")
        return 
    
    syllabus_file = st.sidebar.file_uploader("Upload Syllabus PDF", type="pdf", accept_multiple_files=False)
    question_files = st.sidebar.file_uploader(
        "Upload Question Papers (Min:2, Max:4)", type="pdf", accept_multiple_files=True
    )

    if st.sidebar.button("Run Analysis"):
        if not api_key:
            st.error("Please enter a valid Groq API key to proceed. 🙏🏻")
        if not syllabus_file or not question_files:
            st.error("Please upload both syllabus and question paper PDFs.")
            return

        # Extract text from PDFs
        syllabus_text = extract_text_from_pdf(syllabus_file)
        question_papers = {
            idx: extract_text_from_pdf(file) for idx, file in enumerate(question_files)
        }

        if not syllabus_text or not any(question_papers.values()):
            st.error("Failed to extract text from uploaded files.")
            return

        # Create and run workflow
        workflow = create_exam_analysis_workflow(api_key, syllabus_text, question_papers)
        results = await workflow()

        # Display results
        if results:
            st.success("Analysis complete!")
            st.subheader("Exam Analysis")
            st.text_area("Exam Analysis", results.get("exam_analysis", ""), height=300)

            st.subheader("Practice Questions")
            st.text_area("Practice Questions", results.get("practice_questions", ""), height=300)

            # Download results
            if st.button("Download Analysis"):
                with open("analysis.txt", "w", encoding="utf-8") as f:
                    f.write(results["exam_analysis"])
                st.success("Analysis file created!")

            if st.button("Download Questions"):
                with open("practice.txt", "w", encoding="utf-8") as f:
                    f.write(results["practice_questions"])
                st.success("Questions file created!")

            # Ensure download buttons are only shown after files are created
            if os.path.exists("analysis.txt"):
                st.download_button("Download Exam Analysis", "analysis.txt", file_name="analysis.txt")

            if os.path.exists("practice.txt"):
                st.download_button("Download Practice Questions", "practice.txt", file_name="practice.txt")

# Run Streamlit App
if __name__ == "__main__":
    asyncio.run(main())
