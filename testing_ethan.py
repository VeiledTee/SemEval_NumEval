from transformers import AutoTokenizer, AutoModel
import torch


def get_embeddings(sentence: str, model_name: str):
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize the input sentence
    tokens = tokenizer(sentence, return_tensors="pt")

    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(**tokens)

    # Extract the embeddings from the output
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embeddings


# Example usage
input_sentence = "This is a sample sentence."
bert_embeddings = get_embeddings(input_sentence, "bert-base-uncased")
sentence_bert_embeddings = get_embeddings(input_sentence, "facebook/bart-large-cnn")
bart_embeddings = get_embeddings(input_sentence, "sentence-transformers/paraphrase-MiniLM-L6-v2")

print("BERT embeddings:", bert_embeddings.shape)
print("Sentence-BERT embeddings:", sentence_bert_embeddings.shape)
print("BART embeddings:", bart_embeddings.shape)
