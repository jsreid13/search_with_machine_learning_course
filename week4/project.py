from sentence_transformers import SentenceTransformer, util
from itertools import combinations

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

sentences = ['I am a cat', 'I am a kitten', 'I am not a cat', 'I am a dog', 'I am a hot dog']
embeddings = model.encode(sentences)
for i in combinations(range(len(sentences)), 2):
   print(sentences[i[0]] + ',' + sentences[i[1]] + ': ' + \
           str(util.cos_sim(embeddings[i[0]],embeddings[i[1]])))
