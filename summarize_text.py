from transformers import BartForConditionalGeneration, BartTokenizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load BART model and tokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def summarize_text(input_text, max_length=150):
    try:
        # Tokenize the input text
        inputs = tokenizer(input_text, max_length=max_length, return_tensors='pt', truncation=True)

        # Generate summary
        summary_ids = model.generate(inputs['input_ids'], max_length=50, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)

        # Decode and return summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        logging.error(f"Error in summarization: {e}")
        return "Error: Unable to summarize text"

# Example text to summarize
input_text = """
t's lunchtime - which means it's time for a round-up of the main things you need to know from the Politics Hub.

The new Labour government has had a busy morning, and it could end up being quite the pivotal day for the future of the Tory party too…

    Parliament has returned for the first time since the election, with the Speaker having been re-elected and the swearing in of all MPs under way;
    Sir Keir Starmer welcomed the diversity of the new parliament in his first Commons speech as PM, while Rishi Sunak vowed the Tories would be an "effective and professional" opposition;
    We also heard from the other party leaders, with Nigel Farage taking the opportunity for a dig at ex-Speaker John Bercow for his handling of parliament's debates on Brexit all those years ago.

Starmer praises 'most diverse parliament' in first Commons speech as prime minister

Sky News

    Junior doctors say they've held a "positive" first meeting with the new health secretary about their ongoing pay dispute;
    Representatives from the BMA union said the government was clearly showing "urgency" to resolve the issue and avoid more strikes, but didn't indicate whether a deal was close;
    Our political correspondent Tamara Cohen said there may only be "a matter of weeks" to find an agreement before the union holds a vote on whether to strike further..
"""

# Summarize the text
summary = summarize_text(input_text)
print("Original Text:")
print(input_text)
print("\nSummary:")
print(summary)
