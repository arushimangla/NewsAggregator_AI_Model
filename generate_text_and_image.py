import base64
import boto3
import json
import os
import random

bedrock_client = boto3.client("bedrock-runtime", region_name="us-west-2")

text_model_id = "anthropic.claude-v2"
image_model_id = "stability.stable-diffusion-xl-v1"

# Step 1: Generate Headline and Sentiment
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

# Step 2: Generate Image
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

if __name__ == "__main__":
    news_article = input("Enter the full text of the news article: ")

    print("\nGenerating summary and sentiment...")
    text_response = generate_summary_and_sentiment(news_article)
    print("\nText Response:\n", text_response)

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

    print("\nGenerated Summary:\n", summary)
    print("\nSentiment:\n", sentiment)

    print("\nGenerating image...")
    image_path = generate_image(summary)
    print(f"\nImage saved at: {image_path}")
