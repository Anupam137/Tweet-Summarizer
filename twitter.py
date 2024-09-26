import streamlit as st 
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
import tweepy  # To interact with Twitter API for fetching tweets

# Twitter API credentials
api_key = "YOUR_API_KEY"
api_secret_key = "YOUR_API_SECRET_KEY"
access_token = "YOUR_ACCESS_TOKEN"
access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Model and tokenizer loading
checkpoint = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)

# Function to summarize tweet and extract Twitter handle
def summarize_tweet_with_tag(tweet_text):
    # Extract Twitter handle from tweet text
    twitter_handle = extract_twitter_handle(tweet_text)

    # Prepare input text for summarization
    input_text = f"{tweet_text} (from @{twitter_handle})"

    # Perform summarization
    inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    inputs = inputs.to(device)
    summary_ids = base_model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Function to extract Twitter handle from tweet text
def extract_twitter_handle(tweet_text):
    # Use regex to find the first occurrence of @username
    match = re.search(r'@([a-zA-Z0-9_]+)', tweet_text)
    if match:
        return match.group(1)
    else:
        return "UnknownUser"

# Streamlit app
st.set_page_config(layout="wide")

def main():
    st.title("Tweets Summarizer")

    tweet_link = st.text_input("Enter Tweet Link:")

    if st.button("Summarize"):
        try:
            # Extract tweet ID from the tweet link
            tweet_id = tweet_link.split("/")[-1]
            
            # Fetch tweet using Twitter API
            tweet = api.get_status(tweet_id, tweet_mode="extended")
            tweet_text = tweet.full_text

            # Summarize tweet with tagged account
            summary = summarize_tweet_with_tag(tweet_text)

            # Display summarization
            st.info("Summarization Complete")
            st.success(summary)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
