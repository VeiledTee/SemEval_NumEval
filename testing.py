from huggingBart import BartModel
import numpy as np

modelName = "facebook/bart-base" #this is the name of the hugging face model
model = BartModel(modelName)

testStrings = ["hello world!", 
               "its the end of the world!", 
               "and i feel fine.", 
               "so long and thnaks for all the fish", 
               "so sad it had to come to this."]


embeddings = []
for text in testStrings:
    embeddings.append(model.getEmbeddings(text))

embeddings = np.array(embeddings)
print(f"{embeddings.shape()}")