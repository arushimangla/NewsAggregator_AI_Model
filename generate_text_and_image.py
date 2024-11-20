import base64
import boto3
import json
import os
import random

bedrock_client = boto3.client("bedrock-runtime", region_name="us-west-2")

text_model_id = "anthropic.claude-v2"
image_model_id = "stability.stable-diffusion-xl-v1"

# Step 1: Generate Headline and Sentiment
def generate_headline_and_sentiment(news_article):
    instruction = """I am going to provide you with the full text of a news article. Your task is twofold:
    
    1. Headline Generation: Create a concise, engaging, and accurate headline that captures the main idea or essence of the article. The headline should be no more than 10-12 words and should be suitable for a general audience. If the article has a tone, such as informative, inspiring, or urgent, try to reflect that in the headline as well.
    
    2. Sentiment Analysis: Analyze the overall sentiment of the article and classify it as either positive, negative, or neutral. Additionally, provide a brief explanation for your sentiment classification.

    Here is the article:"""

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
def generate_image(headline):
    prompt = f"Create a cinematic, high-resolution 4K HDR image that visually represents the following headline: '{headline}'. It should convey the tone and theme of the headline."

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

    print("\nGenerating headline and sentiment...")
    text_response = generate_headline_and_sentiment(news_article)
    print("\nText Response:\n", text_response)

    if "Headline:" in text_response:
        headline_start = text_response.find("Headline:") + len("Headline:")
        headline_end = text_response.find("\n", headline_start)
        headline = text_response[headline_start:headline_end].strip()
    else:
        headline = "Headline not found in response."
    print("\nGenerating image...")
    image_path = generate_image(headline)
    print(f"\nImage saved at: {image_path}")
