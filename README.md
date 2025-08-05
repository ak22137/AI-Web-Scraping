# AI-Web-Scraping

## Overview
AI-Web-Scraping is an agentic, human-in-the-loop workflow for automated book publication and content improvement. It scrapes web content from a given URL, uses Gemini LLM to spin and improve chapters, and leverages reinforcement learning (RL) to iteratively enhance output based on human feedback. All versions are saved in ChromaDB for versioning and search.

## Features
- **Web Scraping & Screenshot:** Uses Playwright to fetch chapter text and save a screenshot of the source page.
- **AI Writing & Review:** Gemini LLM spins and improves the scraped content.
- **Human-in-the-Loop:** Human reviewers can approve, edit, or reject AI output, guiding the workflow interactively.
- **RL-Based Improvement:** A simple RL agent learns from human feedback and text features to suggest better improvements in future iterations.
- **Versioning:** Each improved version is saved in ChromaDB for tracking and semantic search.

## Workflow
1. Scrape chapter text and screenshot from a given URL.
2. Use Gemini LLM to generate a spun/improved version.
3. Human reviews the output and provides feedback (`approve`, `edit`, `reject`, or custom text).
4. RL agent calculates a reward and updates its learning to improve future suggestions.
5. Save each version to ChromaDB.
6. Repeat for up to 3 iterations or until approved.

## Technologies Used
- Python
- Playwright (scraping & screenshots)
- Gemini LLM (via google-generativeai)
- ChromaDB (versioning & semantic search)
- RL agent (Q-learning, numpy)

## How to Run
1. Clone the repository.
2. Create a Python virtual environment (`python -m venv venv`).
3. Install dependencies (`pip install -r requirements.txt`).
4. Add your Gemini API key to `.env`.
5. Run `main.py` and follow the prompts for human review.

## License
Developer retains their license. This project is for evaluation purposes only.
