import os
import numpy as np
import json
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from playwright.sync_api import sync_playwright
from chromadb import Client as ChromaClient

# Load Gemini API key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("")
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

def ai_writer(text, rl_agent=None, iteration=0):
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Use RL agent to improve prompt if available
    if rl_agent and iteration > 0:
        text = rl_based_improvement(text, rl_agent)
    
    prompt = f"Spin this chapter: {text}"
    response = model.generate_content(prompt)
    return response.text

def human_review(text):
    print("AI Output:\n", text[:1000])  # Print first 1000 chars for brevity
    print("\nOptions: 'approve', 'edit', 'reject', or type new text")
    user = input("Your feedback: ").strip().lower()
    
    if user in ['approve', 'edit', 'reject']:
        return text, user
    elif user:
        return user, 'edit'
    else:
        return text, 'approve'

def save_version(text, db):
    db.add(
        ids=[str(db.count() + 1)],
        documents=[text]
    )

def rl_reward_stub(text):
    # Placeholder for RL-based reward logic
    print("[RL Reward Model] Reward calculated (stub).")
    return 1.0

class SimpleRLAgent:
    """Simple RL agent for text quality assessment"""
    
    def __init__(self):
        self.q_table = {}  # State-action values
        self.learning_rate = 0.1
        self.epsilon = 0.1  # Exploration rate
        self.history_file = "rl_history.json"
        self.load_history()
    
    def get_text_features(self, text):
        """Extract simple features from text"""
        words = text.split()
        return {
            'length': len(text),
            'word_count': len(words),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'sentence_count': text.count('.') + text.count('!') + text.count('?'),
            'paragraph_count': text.count('\n\n') + 1
        }
    
    def get_state(self, text):
        """Convert text to discrete state"""
        features = self.get_text_features(text)
        # Discretize features into bins
        length_bin = min(4, features['length'] // 500)  # 0-4 bins
        word_bin = min(4, features['word_count'] // 100)  # 0-4 bins
        return f"len_{length_bin}_words_{word_bin}"
    
    def get_reward(self, text, human_feedback=None):
        """Calculate reward based on text quality and human feedback"""
        features = self.get_text_features(text)
        
        # Base reward from text metrics
        base_reward = 0.0
        
        # Reward for reasonable length (500-2000 chars is good)
        if 500 <= features['length'] <= 2000:
            base_reward += 0.3
        elif features['length'] < 200:
            base_reward -= 0.2
        
        # Reward for good word count (50-300 words is good)
        if 50 <= features['word_count'] <= 300:
            base_reward += 0.2
        
        # Reward for reasonable average word length (4-8 chars)
        if 4 <= features['avg_word_length'] <= 8:
            base_reward += 0.1
        
        # Reward for proper sentence structure
        if features['sentence_count'] > 0:
            base_reward += 0.1
        
        # Human feedback (most important)
        if human_feedback is not None:
            if human_feedback == "approve":
                base_reward += 0.5
            elif human_feedback == "edit":
                base_reward -= 0.3
            elif human_feedback == "reject":
                base_reward -= 0.5
        
        return np.clip(base_reward, -1.0, 1.0)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        actions = ["improve_length", "improve_style", "improve_structure", "keep_as_is"]
        
        if np.random.random() < self.epsilon:
            return np.random.choice(actions)
        
        # Choose best action based on Q-values
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in actions}
        
        return max(self.q_table[state], key=self.q_table[state].get)
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning"""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in ["improve_length", "improve_style", "improve_structure", "keep_as_is"]}
        
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in ["improve_length", "improve_style", "improve_structure", "keep_as_is"]}
        
        # Q-learning update
        max_next_q = max(self.q_table[next_state].values())
        self.q_table[state][action] += self.learning_rate * (reward + 0.9 * max_next_q - self.q_table[state][action])
    
    def save_history(self):
        """Save Q-table and learning history"""
        history = {
            'q_table': self.q_table,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_history(self):
        """Load previous Q-table if exists"""
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
                self.q_table = history.get('q_table', {})
                print(f"[RL] Loaded Q-table with {len(self.q_table)} states")
        except FileNotFoundError:
            print("[RL] No previous history found, starting fresh")

def rl_based_improvement(text, rl_agent):
    """Use RL agent to suggest improvements"""
    state = rl_agent.get_state(text)
    action = rl_agent.select_action(state)
    
    print(f"[RL Agent] Current state: {state}")
    print(f"[RL Agent] Suggested action: {action}")
    
    # Apply action-based prompts
    if action == "improve_length":
        return f"Make this text more detailed and elaborate: {text}"
    elif action == "improve_style":
        return f"Improve the writing style and flow of: {text}"
    elif action == "improve_structure":
        return f"Improve the structure and organization of: {text}"
    else:
        return text

def main():
    url = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
    print("Scraping chapter and taking screenshot...")
    chapter = scrape_chapter(url)

    print("Chapter scraped. Initializing RL Agent...")
    rl_agent = SimpleRLAgent()
    
    print("Starting RL-based iterative improvement...")
    current_text = chapter
    max_iterations = 3
    
    for iteration in range(max_iterations):
        print(f"\n=== Iteration {iteration + 1} ===")
        
        # Get current state
        current_state = rl_agent.get_state(current_text)
        
        # AI processing with RL guidance
        print("Processing with Gemini + RL guidance...")
        spun = ai_writer(current_text, rl_agent, iteration)
        
        # Human review with feedback
        print("Reviewing with human-in-the-loop...")
        final_text, feedback = human_review(spun)
        
        # Calculate reward
        reward = rl_agent.get_reward(final_text, feedback)
        print(f"[RL] Reward: {reward:.2f}")
        
        # Get next state and update Q-values
        next_state = rl_agent.get_state(final_text)
        if iteration > 0:  # We need a previous action to update
            rl_agent.update_q_value(prev_state, prev_action, reward, current_state)
        
        # Store current state and action for next update
        prev_state = current_state
        prev_action = rl_agent.select_action(current_state)
        
        # Save version to ChromaDB
        print("Saving version to ChromaDB...")
        db = ChromaClient().get_or_create_collection("chapters")
        save_version(final_text, db)
        
        # Update current text for next iteration
        current_text = final_text
        
        # Check if we should stop 
        if feedback == 'approve':
            print("User approved! Stopping iterations.")
            break
    
    # Save RL learning 
    rl_agent.save_history()
    print(f"\nWorkflow complete! Screenshot saved as screenshot.png.")
    print(f"Final reward: {reward:.2f}")
    print(f"RL Agent learned from {len(rl_agent.q_table)} states.")

if __name__ == "__main__":
    main()
