import os
import sys
from dotenv import load_dotenv

# Ensure we can import from src
sys.path.append(os.getcwd())

load_dotenv()

from fastapi.testclient import TestClient
from src.api.app import create_app

app = create_app()
client = TestClient(app)

def test_spec_endpoint():
    print("\n--- Testing /test/spec ---")
    ticket_text = """
                ## 📝 Overview
                Currently, the calculator only runs in the terminal. This issue tracks the development of a graphical user interface (GUI) to make the tool more accessible.

                ## 🎯 User Story
                **As a** user,
                **I want** to interact with a visual calculator interface,
                **So that** I don't have to use the command line for basic arithmetic.

                ## 🛠 Technical Requirements
                * **Logic Integration:** Import the existing calculation engine from `src/logic/` (do not rewrite the math).
                * **Framework:** React (using Functional Components).
                * **Styling:** Tailwind CSS (Grid layout for the keypad).
                * **State:** Use `useState` to manage the display and operation string.

                ## 📋 Task List
                - [ ] Create a `Calculator.js` component.
                - [ ] Implement a 4x4 grid for numbers and operators.
                - [ ] Connect "Click" events to the existing `calculate()` function.
                - [ ] Add a display area for the current input and result.
                - [ ] Add a "Clear" button functionality.

                ## ✅ Acceptance Criteria
                - [ ] Interface is responsive (works on mobile and desktop).
                - [ ] Result matches CLI output for the same operations.
                - [ ] No console errors during execution.
                - [ ] Background is "Dark Mode" themed.
                """
    payload = {
        "state": {
            "ticket_text": ticket_text
        },
        "mock_files": {
            "src/logic/calculator.py": "def calculate(expression): return eval(expression)"
        },
        "cleanup": True
    }
    response = client.post("/test/spec", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Agent Status: {data['status']}")
        print(f"Summary: {data['summary']}")
        print(f"Spec length: {len(data['state_updates'].get('spec', ''))}")
    else:
        print(f"Error: {response.text}")

def test_coding_endpoint():
    print("\n--- Testing /test/coding ---")
    payload = {
        "state": {
            "spec": "Implement the calculator GUI using React and Tailwind CSS.",
            "detected_language": "Python",
            "detected_framework": "React",
            "detected_styling": "Tailwind CSS",
        },
        "mock_files": {
            "calculator.py": "def add(a, b): return a + b\n\ndef subtract(a, b): return a + b"
        },
        "cleanup": False # Keep it to verify file content manually if needed
    }
    response = client.post("/test/coding", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Agent Status: {data['status']}")
        print(f"Summary: {data['summary']}")
        print(f"Workspace: {data['workspace_path']}")
        
        # Verify the file was changed
        workspace_path = data['workspace_path']
        calc_path = os.path.join(workspace_path, "calculator.py")
        if os.path.exists(calc_path):
            with open(calc_path, "r") as f:
                content = f.read()
                print("Updated calculator.py content:")
                print(content)
                if "a - b" in content:
                    print("SUCCESS: File updated correctly.")
                else:
                    print("FAILURE: File not updated correctly.")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_spec_endpoint()
    test_coding_endpoint()
