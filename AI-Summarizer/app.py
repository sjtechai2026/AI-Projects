from flask import Flask, render_template, request, jsonify
import requests
import subprocess
import time

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"


def get_models():
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )

        lines = result.stdout.splitlines()[1:]

        models = []

        for line in lines:
            if line.strip():
                model = line.split()[0]
                models.append(model)

        return models

    except Exception:
        return ["llama3.2:latest"]


@app.route("/")
def home():
    models = get_models()

    return render_template(
        "index.html",
        models=models
    )


@app.route("/summarize", methods=["POST"])
def summarize():

    try:

        data = request.get_json()

        text = data.get("text", "")
        model = data.get(
            "model",
            "llama3.2:latest"
        )

        prompt = f"""
Summarize the following text in 3 bullet points:

{text}
"""

        start = time.time()

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        result = response.json()

        end = time.time()

        summary = result.get(
            "response",
            "No summary generated."
        )

        return jsonify({

            "summary": summary,

            "stats": {

                "model": model,

                "input_chars": len(text),

                "output_chars": len(summary),

                "estimated_input_tokens":
                    len(text) // 4,

                "response_time":
                    round(end - start, 2)

            }

        })

    except Exception as e:

        return jsonify({

            "summary":
                f"Error: {str(e)}",

            "stats": {

                "model": "-",

                "input_chars": 0,

                "output_chars": 0,

                "estimated_input_tokens": 0,

                "response_time": 0

            }

        })


if __name__ == "__main__":
    app.run(debug=True)