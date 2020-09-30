from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import BartForSequenceClassification, BartTokenizer

from torch.nn import functional as F
import numpy as np
import pandas as pd
from ar_en_translation import ar_en_translation
sentence = 'The Sports Activity Department of the Deanship of Student Affairs organizes an indoor championship for table tennis and for all departments of the College of Engineering, for males and females separately, on Tuesday 19/2/2019, at 12:00 in College of Science B5.'

# run inputs through model and mean-pool over the sequence
# dimension to get sequence-level representations
# inputs = tokenizer.batch_encode_plus([sentence] + labels,
#                                      return_tensors='pt',
#                                      pad_to_max_length=True)
# input_ids = inputs['input_ids']
# attention_mask = inputs['attention_mask']
# output = model(input_ids, attention_mask=attention_mask)[0]
# sentence_rep = output[:1].mean(dim=1)
# label_reps = output[1:].mean(dim=1)
#
# # now find the labels with the highest cosine similarities to
# # the sentence
# similarities = F.cosine_similarity(sentence_rep, label_reps)
# closest = similarities.argsort(descending=True)
# for ind in closest:
#     print(f'label: {labels[ind]} \t similarity: {similarities[ind]}')

###############

premise = sentence = 'The Sports Activity Department of the Deanship of Student Affairs organizes an indoor championship for table tennis and for all departments of the College of Engineering, for males and females separately, on Tuesday 19/2/2019, at 12:00 in College of Science B5.'
# labels = ['science', 'sport', 'education']
# for i in range(len(labels)):
#     hypothesis = f'This text is about {labels[i]}.'
#
#     # run through model pre-trained on MNLI
#     x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
#                          max_length=tokenizer.max_len,
#                          truncation_strategy='only_first')
#     logits = model(x.to('cpu'))[0]
#
#     # we throw away "neutral" (dim 1) and take the probability of
#     # "entailment" (2) as the probability of the label being true
#     entail_contradiction_logits = logits[:,[0,2]]
#     probs = entail_contradiction_logits.softmax(1)
#     prob_label_is_true = probs[:,1]
#     print("{0} probabilty is {1}".format(labels[i], prob_label_is_true))

##############
# load model pretrained on MNLI
# tokenizer = BartTokenizer.from_pretrained('bart-large-mnli')
# model = BartForSequenceClassification.from_pretrained('bart-large-mnli')

tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

# pose sequence as a NLI premise and label (politics) as a hypothesis
premise = 'The Sports Activity Department of the Deanship of Student Affairs organizes an indoor championship for table tennis and for all departments of the College of Engineering, for males and females separately, on Tuesday 19/2/2019, at 12:00 in College of Science B5.'
hypothesis = 'This text is about education.'

def prob_label(premise, label):
    # run through model pre-trained on MNLI
    hypothesis = 'This text is about {}.'.format(label)
    input_ids = tokenizer.encode(premise, hypothesis, return_tensors='pt', max_length=tokenizer.max_len, truncation_strategy='only_first')

    logits = model(input_ids)[0]

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true
    entail_contradiction_logits = logits[:,[0,2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    # true_prob = probs[:,1].item() * 100
    # print(f'Probability that the label is true: {probs[:,1].item()* 100:0.2f}%')
    return probs[:,1].item()* 100

# def print_similarities(sentences, labels):
#     for i in range(len(sentences)):
#         similarities=[]
#         for j in range(len(labels)):
#             similarities.append(prob_label(sentences[i], labels[j]))
#         similarities = np.array(similarities)
#         similarities_arg_sort = np.argsort(-similarities)
#         # closest = -np.sort(-similarities)
#         print(sentences[i])
#         for ind in similarities_arg_sort:
#             # print(f'label: {labels[ind]} \t Probability that the label is true: {similarities[ind]:0.2f}%')
#     return sentences

def return_similarities(sentences):
    sents_similarities = []
    for i in range(len(sentences)):
        similarities, sent_similarity_sort=[], []
        for j in range(len(labels)):
            similarities.append(prob_label(sentences[i], labels[j]))
        similarities = np.array(similarities)
        similarities_arg_sort = np.argsort(-similarities)
        # closest = -np.sort(-similarities)
        # print(sentences[i])
        for ind in similarities_arg_sort:
        #     print(f'label: {labels[ind]} \t Probability that the label is true: {similarities[ind]:0.2f}%')
            sent_similarity_sort.append(similarities[ind])
        sents_similarities.append(sent_similarity_sort)
    return sents_similarities


labels = ['soccer', 'programming', 'sport', 'education', 'jewish', 'israel', 'palestine', 'islam', 'football', 'health care', 'movies']
# print_similarities(ar_en_translation('نزلت ترشيحات جوائز التوته الذهبيه لهذي السنه واللي تمنح لاسوأ اعمال هوليوود سنويا والمرشحين لجائزة اسوأ فيلم هم'), labels)
# print('\n')
# print_similarities(['Nominations for the Golden Raspberry Awards were awarded for this year, which are awarded to the worst acts of Hollywood annually, and the candidates for the award for the worst movie are'], labels)
