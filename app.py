from flask import Flask, jsonify, request
import base64
import boto3
import json
import os
import random

app = Flask(__name__)

bedrock_client = boto3.client("bedrock-runtime", region_name="us-west-2")

# Models
text_model_id = "anthropic.claude-v2"
image_model_id = "stability.stable-diffusion-xl-v1"

# Utility Functions
def generate_summary_and_sentiment(news_article):
    instruction = """I am going to provide you with the topic or a brief description of a news article. Your task is to:

    1. Summary Generation: Write a concise, clear, and accurate summary of the article in EXACTLYgit add  8-9 lines. The summary should include key points, cover the main aspects of the topic, and provide essential details. Maintain a neutral tone and avoid overly technical or verbose language. Ensure the summary is easy to understand and captures the essence of the article.

    2. Sentiment Analysis: Analyze the overall sentiment of the topic and classify it as positive, negative, or neutral. Provide a brief explanation for your sentiment classification.

    The output format should be:
    Summary:
    Sentiment: (Positive, Negative or Neutral)

    Here is the topic or description:"""

    prompt_data = f"{instruction} {news_article}"

    payload = {
        "prompt": f"\n\nHuman:{prompt_data}\n\nAssistant:",
        "max_tokens_to_sample": 512,
        "temperature": 0.8,
        "top_p": 0.8,
    }

    response = bedrock_client.invoke_model(
        body=json.dumps(payload),
        modelId=text_model_id,
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    return response_body.get("completion")


def generate_image(title):
    prompt = f"Create a cinematic, high-resolution 4K HDR image that visually represents the following description: '{title}'. It should convey the tone and theme of the description."

    seed = random.randint(0, 4294967295)

    native_request = {
        "text_prompts": [{"text": prompt}],
        "style_preset": "photographic",
        "seed": seed,
        "cfg_scale": 10,
        "steps": 30,
    }

    response = bedrock_client.invoke_model(
        body=json.dumps(native_request),
        modelId=image_model_id,
        accept="application/json",
        contentType="application/json",
    )

    model_response = json.loads(response["body"].read())
    base64_image_data = model_response["artifacts"][0]["base64"]

    i, output_dir = 1, "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    while os.path.exists(os.path.join(output_dir, f"generated_image_{i}.png")):
        i += 1

    image_path = os.path.join(output_dir, f"generated_image_{i}.png")
    with open(image_path, "wb") as file:
        file.write(base64.b64decode(base64_image_data))

    return image_path


# API Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "Service is running"})


@app.route('/generate', methods=['POST'])
def generate_content():
    data = request.get_json()
    if not data or 'news_article' not in data:
        return jsonify({"error": "Invalid input. 'news_article' is required."}), 400

    news_article = data['news_article']

    try:
        text_response = generate_summary_and_sentiment(news_article)
    except Exception as e:
        return jsonify({"error": f"Text generation failed: {str(e)}"}), 500

    if "Summary:" in text_response:
        summary_start = text_response.find("Summary:") + len("Summary:")
        summary_end = text_response.find("Sentiment:")  # Assuming sentiment starts after the summary
        summary = text_response[summary_start:summary_end].strip()
    else:
        summary = "Summary not found in response."

    if "Sentiment:" in text_response:
        sentiment_start = text_response.find("Sentiment:") + len("Sentiment:")
        sentiment = text_response[sentiment_start:].strip()
    else:
        sentiment = "Sentiment not found in response."

    try:
        image_path = generate_image(summary)
    except Exception as e:
        return jsonify({"error": f"Image generation failed: {str(e)}"}), 500

    return jsonify({
        "summary": summary,
        "sentiment": sentiment,
        "image_path": image_path
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
