import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')


def paraphrase_bart(test):
    def generate(model, tokenizer, texts):
        ans = []
        print("Bart generate answers:")
        for i in tqdm(range(len(texts))):
            prompt = texts[i]
            input_ = "paraphrase to be nontoxic: \n" + prompt
            input_ids = tokenizer(input_, return_tensors="pt").input_ids
            outputs = model.generate(input_ids=input_ids)
            ans.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        return ans
    
    model = AutoModelForSeq2SeqLM.from_pretrained("./models/bart")
    tokenizer = AutoTokenizer.from_pretrained("./models/bart")
    model.eval()
    model.config.use_cache = False
    test['generated'] = generate(model, tokenizer, test['toxic'].to_list())
    test.head()
    test.to_csv("./data/interim/bart_pred.csv")

def paraphrase_t5(test):
    def generate(model, tokenizer, texts):
        ans = []
        print("T5 generate answers:")
        for i in tqdm(range(len(texts))):
            prompt = texts[i]
            input_ = "detoxify: "+ prompt + ' </s>'
            input_ids = tokenizer.batch_encode_plus(
                        [input_], return_tensors="pt"
                    ).input_ids
            outputs = model.generate(input_ids=input_ids)
            ans.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        return ans

    model = T5ForConditionalGeneration.from_pretrained("./models/t5")
    model.eval()
    model.config.use_cache = False
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    test['generated'] = generate(model, tokenizer, test['toxic'].to_list())
    test.to_csv("./data/interim/t5_pred.csv")

test = pd.read_csv("./data/raw/test.csv", index_col=0)
paraphrase_bart(test)
paraphrase_t5(test)
