from speechbrain.inference.diarization import Speech_Emotion_Diarization
import os
from pandas import read_csv

# model webpage: https://huggingface.co/speechbrain/emotion-diarization-wavlm-large
# to run on GPU, add `run_opts={"device": "cuda"}` when calling from_hparams
# if computing node does not have access to internet, save classifier using torch.save(path/to/saved/model), 
# and then torch.load(path/to/saved/model), then use diarize_file as usual
# diarize_file will automatically check sampling rate and resample

run_opts={"device": "cuda:3"}
classifier = Speech_Emotion_Diarization.from_hparams(
    source="speechbrain/emotion-diarization-wavlm-large", run_opts=run_opts
)
language = ["en", "cn", "de"]
function = ["expectation", "question", "negation"]
root_path = os.getcwd()

# test cn, de, en (excluding tianjin)
for lang in language:
    print(f"Diarising language {lang}...")
    path = os.path.join(root_path, lang)
    os.chdir(path)

    wav_df = read_csv("metadata.csv")
    wav_happy = wav_df[wav_df['answer'] == "Happy"]
    wav_sad = wav_df[wav_df['answer'] == "Sad"]
    wav_angry = wav_df[wav_df['answer'] == "Angry"]
    emotion = {"Happy": wav_happy, "Sad": wav_sad, "Angry": wav_angry}

    for emo, df in emotion.items():
        print(f"Diarising emotion {emo}...")
        
        for _, row in df.iterrows():
            wav = row['file']
            wav_path = os.path.join(path, wav)

            with open(f"{emo}_diarise_result.txt", "a") as file:
                diary = classifier.diarize_file(wav_path)
                result = [str(r) for r in diary[wav_path]]
                result = ", ".join(result)
                print(result)
                wav_info = wav + ", " + emo + ":"
                file.write(wav_info)
                file.write("\n")
                file.write(result)
                file.write("\n")

# test tj on sed task
print(f"Diarising language: TJ...")
path = os.path.join(root_path, "tj")
os.chdir(path)

wav_df = read_csv("metadata.csv")
wav_ques = wav_df[wav_df['answer'] == "Question"]
wav_neg = wav_df[wav_df['answer'] == "Negation"]
wav_exp = wav_df[wav_df['answer'] == "Expectation"]
func = {"Question": wav_ques, "Negation": wav_neg, "Expectation": wav_exp}

for f, df in func.items():
    print(f"Diarising speech labelled function: {f}...")
    
    for _, row in df.iterrows():
        wav = row['file']
        wav_path = os.path.join(path, wav)

        with open(f"{f}_diarise_result.txt", "a") as file:
            diary = classifier.diarize_file(wav_path)
            result = [str(r) for r in diary[wav_path]]
            result = ", ".join(result)
            print(result)
            wav_info = wav + ", " + f + ":"
            file.write(wav_info)
            file.write("\n")
            file.write(result)
            file.write("\n")

# output will be like:
# 0002_000912.wav, Happy:
# {'start': 0.0, 'end': 2.42, 'emotion': 'h'}, {'start': 2.42, 'end': 2.7, 'emotion': 'a'}
