from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


def init():
    """
    :return: Tokenizer, model
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens"), \
           AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens").to(device), \
           device


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def calculate_embeddings(tokenizer, model, device, sentences: list = None) -> np.array:
    if not sentences:
        sentences = ['This framework generates embeddings for each input sentence']

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input.to(device))

    # Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
    return sentence_embeddings


if __name__ == '__main__':
    tok, mod, dev = init()
    calculate_embeddings(tok, mod, dev)
