import pandas as pd
from pathlib import Path
from transliterate import translit
from deep_translator import GoogleTranslator
from scipy.stats import pearsonr, spearmanr
from scipy.special import logsumexp
import asyncio
import numpy as np

async def translate_text(text, src="ka", dest="en"):
    return GoogleTranslator(source=src, target=dest).translate(text)

async def translate_column(data, column_name, result_column_name):
    translated_values = []
    for text in data[column_name]:
        if isinstance(text, str):
            try:
                translated_text = await translate_text(text)
                translated_values.append(translated_text)
            except:
                translated_values.append("X")
        else:
            translated_values.append(text)
    data[result_column_name] = translated_values
    return data

def analyse_sentences(file_path, sort_by='certainty', ascending=True):
    data = pd.read_csv(file_path)
    # data = data.iloc[:3,]
    
    eps = 1e-12  # prevents log(0)

    logp_erg = np.log(data["p_form_grammatical_erg"] + eps)
    logp_nom = np.log(data["p_form_ungrammatical_nom"] + eps)
    logp_dat = np.log(data["p_form_ungrammatical_dat"] + eps)

    data["log_odds"] = logp_erg - logsumexp(
        np.vstack([logp_nom, logp_dat]).T,
        axis=1
    )

    data["word_length"] = data["form_grammatical_erg"].apply(lambda word: len(word))
    data["sent_length"] = data["masked_text"].apply(lambda sent: len(sent))

    pearson_corr_word, _ = pearsonr(data["word_length"],data["log_odds"])
    spearman_corr_word, _ = spearmanr(data["word_length"],data["log_odds"])

    pearson_corr_sent, _ = pearsonr(data["sent_length"],data["log_odds"])
    spearman_corr_sent, _ = spearmanr(data["sent_length"],data["log_odds"])

    data['translit'] = data['form_grammatical_erg'].apply(
        lambda x: translit(x, 'ka', reversed=True) if isinstance(x, str) else x
    )

    sorted_data = data.sort_values(by=sort_by, ascending=ascending)

    result = sorted_data[['sentence_id', 'masked_text', 'form_grammatical_erg',
                           'form_ungrammatical_nom', 'form_ungrammatical_dat',
                           'p_form_grammatical_erg', 'p_form_ungrammatical_nom',
                           'p_form_ungrammatical_dat', 'certainty','log_odds', 'translit']]

    return result, pearson_corr_word, spearman_corr_word, pearson_corr_sent, spearman_corr_sent

async def main():
    RESULTS_PATH = "./results/full"
    file_path = Path(RESULTS_PATH) / 'mlm-mBERT-S2-ERG-word-level.csv'
    result, pearson_corr_word, spearman_corr_word, pearson_corr_sent, spearman_corr_sent = analyse_sentences(
            file_path, sort_by='log_odds', ascending=True
    )
    result = await translate_column(result, 'form_grammatical_erg','translated_form')
    result = await translate_column(result, 'masked_text','translated_sent')
    for i, row in result.iterrows():
        print("====")
        print("Form: ", row["form_grammatical_erg"])
        print("Rough Translation (Word): ", row["translated_form"])
        print("Rough Translation (Sent): ", row["translated_sent"])
        print("Transliterated: ", row["translit"])
        print("Pref for Erg: ", row["log_odds"])
        print("====")

    print([["form_grammatical_erg","log_odds","translated_form","translit","translated_sent"]])
    print(f"Pearson r (Word): {pearson_corr_word}")
    print(f"Spearman r (Word): {spearman_corr_word}")
    print(f"Pearson r (Sent): {pearson_corr_sent}")
    print(f"Spearman r (Sent): {spearman_corr_sent}")

if __name__ == "__main__":
    asyncio.run(main())
