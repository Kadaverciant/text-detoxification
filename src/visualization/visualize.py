import pandas as pd
import evaluate
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def measure_toxicity(texts, batch_size = 64):
    labels = []
    tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
    model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = tokenizer(texts[i:i + batch_size], return_tensors='pt', padding=True)
        labels.extend(model(**batch)['logits'].argmax(1).float().data.tolist())
    return labels

def calculate_metrics(references, predictions):
    bleu_scores = []
    meteor_scores = []
    
    for i in tqdm(range(len(references))):
        inputs = [predictions[i]]
        refs = [references[i]]

        bleu_score = bleu_metric.compute(predictions=inputs, references=refs)
        bleu_scores.append(bleu_score["bleu"])

        meteor_score = meteor_metric.compute(predictions=inputs, references=refs)
        meteor_scores.append(meteor_score["meteor"])
        
    return bleu_scores, meteor_scores

def drow_toxicity_plot(tl_toxic, tl_nontoxic, tl_bart, tl_t5):
    plt.clf()
    plt.close()

    data = {'input text':tl_toxic, 'reference text':tl_nontoxic, 'bart answers':tl_bart, 't5 answers':tl_t5}
    names = list(data.keys())
    values = list(data.values())
    
    fig = plt.figure(figsize = (10, 5))
    bar_colors = ['red', 'blue', 'orange', 'green']
    
    plt.bar(names, values, color = bar_colors, width = 0.4)

    plt.ylabel("toxicity level")
    plt.title("Toxicity level comparison")
    plt.tight_layout()
    plt.savefig("./reports/figures/toxicity_levels.png")

def drow_bleu_plot(bart_bleu, t5_bleu):
    plt.clf()
    plt.close()

    data = {'bart answers':bart_bleu, 't5 answers':t5_bleu}
    names = list(data.keys())
    values = list(data.values())
    
    fig = plt.figure(figsize = (6, 5))
    bar_colors = ['orange', 'green']
    
    plt.bar(names, values, color = bar_colors, width = 0.4)

    plt.ylabel("score")
    plt.title("BLEU score")
    plt.tight_layout()
    plt.savefig("./reports/figures/bleu_score.png")

def drow_meteor_plot(bart_meteor, t5_meteor):
    plt.clf()
    plt.close()
    
    data = {'bart answers':bart_meteor, 't5 answers':t5_meteor}
    names = list(data.keys())
    values = list(data.values())
    
    fig = plt.figure(figsize = (6, 5))
    bar_colors = ['orange', 'green']
    
    plt.bar(names, values, color = bar_colors, width = 0.4)

    plt.ylabel("score")
    plt.title("METEOR score")
    plt.tight_layout()
    plt.savefig("./reports/figures/meteor_score.png")

df = pd.read_csv("./data/raw/test.csv", index_col=0)
df_t5 = pd.read_csv("./data/interim/t5_pred.csv", index_col=0)
df_bart = pd.read_csv("./data/interim/bart_pred.csv", index_col=0)
df_t5 = pd.read_csv("./data/interim/t5_pred.csv", index_col=0) 
df['bart'] = df_bart['generated']
df['t5'] = df_t5['generated']

df['toxic_label'] = measure_toxicity(df['toxic'].to_list())
df['non-toxic_label'] = measure_toxicity(df['non-toxic'].to_list())
df['bart_label'] = measure_toxicity(df['bart'].to_list())
df['t5_label'] = measure_toxicity(df['t5'].to_list())

tl_toxic = df['toxic_label'][df['toxic_label']==1].count()/len(df)
tl_nontoxic = df['non-toxic_label'][df['non-toxic_label']==1].count()/len(df)
tl_bart = df['bart_label'][df['bart_label']==1].count()/len(df)
tl_t5 = df['t5_label'][df['t5_label']==1].count()/len(df)

drow_toxicity_plot(tl_toxic, tl_nontoxic, tl_bart, tl_t5)

bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")

df['bart_bleu'], df['bart_meteor'] = calculate_metrics(df['toxic'].to_list(), df['bart'].to_list())
df['t5_bleu'], df['t5_meteor'] = calculate_metrics(df['toxic'].to_list(), df['t5'].to_list())
bart_bleu, bart_meteor = df['bart_bleu'].mean(), df['bart_meteor'].mean() 
t5_bleu, t5_meteor = df['t5_bleu'].mean(), df['t5_meteor'].mean() 

drow_bleu_plot(bart_bleu, t5_bleu)
drow_meteor_plot(bart_meteor, t5_meteor)
