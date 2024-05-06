import streamlit as st 
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import re
import base64

# Model and tokenizer loading
checkpoint = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)

# Function to summarize text and extract Twitter handle
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

    uploaded_file = st.file_uploader("Upload your PDF or TXT file", type=['pdf', 'txt'])

    if uploaded_file is not None:
        if st.button("Summarize"):
            if uploaded_file.type == 'text/plain':  # Text file (.txt)
                text_content = uploaded_file.getvalue().decode("utf-8")
                summary = summarize_tweet_with_tag(text_content)
                st.info("Summarization Complete")
                st.success(summary)
            elif uploaded_file.type == 'application/pdf':  # PDF file (.pdf)
                # Implement PDF processing and summarization logic as needed
                st.warning("PDF file summarization is not yet implemented. Please upload a text file (.txt).")

if __name__ == "__main__":
    main()
