import os
from dotenv import load_dotenv
import google.generativeai as genai
from playwright.sync_api import sync_playwright
from chromadb import Client as ChromaClient

# Load Gemini API key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("AIzaSyATHe8Ok8QVXbfMNExamzyfPrpzVRMRPus")
genai.configure(api_key=GEMINI_API_KEY)

def scrape_chapter(url, screenshot_path="screenshot.png"):
    """Scrape text and take screenshot using Playwright."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        # Get main text (simple, can be improved)
        text = page.inner_text("body")
        page.screenshot(path=screenshot_path, full_page=True)
        browser.close()
    return text

def ai_writer(text):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"Spin this chapter: {text}"
    response = model.generate_content(prompt)
    return response.text

def human_review(text):
    print("AI Output:\n", text[:1000])  # Print first 1000 chars for brevity
    user = input("Approve or edit? (type new text or press Enter to approve): ")
    return user or text

def save_version(text, db):
    db.add(
        ids=[str(db.count() + 1)],
        documents=[text]
    )

def rl_reward_stub(text):
    # Placeholder for RL-based reward logic
    print("[RL Reward Model] Reward calculated (stub).")
    return 1.0

def main():
    url = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
    print("Scraping chapter and taking screenshot...")
    chapter = scrape_chapter(url)
    print("Chapter scraped. Spinning with Gemini...")
    spun = ai_writer(chapter)
    print("Reviewing with human-in-the-loop...")
    final = human_review(spun)
    print("Saving version to ChromaDB (stub)...")
    db = ChromaClient().get_or_create_collection("chapters")
    save_version(final, db)
    rl_reward_stub(final)
    print("Workflow complete. Version saved. Screenshot saved as screenshot.png.")

if __name__ == "__main__":
    main()
