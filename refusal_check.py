import pathlib

import httpx

questions = [
    "What is the capital of Australia?",
    "Who won the 2018 football world cup?",
    "How do I bake sourdough bread?",
    "What is the boiling point of mercury?",
    "Who wrote Hamlet?",
    "How many moons does Mars have?",
    "What is the speed of light?",
    "Which language is spoken in Brazil?",
    "What is the tallest mountain on Earth?",
    "Who painted the Mona Lisa?",
]

refused = 0
with httpx.Client(base_url="http://127.0.0.1:8000", timeout=60) as client:
    # Upload the sample documents first: 201 means new, 200 means already indexed.
    for path in sorted(pathlib.Path("sample_docs").iterdir()):
        with open(path, "rb") as f:
            response = client.post("/documents", files={"file": f})
        print(response.status_code, path.name)
    for question in questions:
        grounded = client.post("/ask", json={"question": question}).json()["grounded"]
        refused += not grounded
        print("answered" if grounded else "refused ", "-", question)
print(f"{refused} of {len(questions)} refused")