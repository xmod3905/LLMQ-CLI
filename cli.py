print("Tunggu Sebentar!")
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import argparse, os 

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

try:
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
except OSError:
    print(f"Model not found locally. Downloading {model_name}...")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def clean_text(text):
    """
    Removes special characters from the given text
    """
    cleaned_text = ""
    for char in text:
        if char.isalnum() or char.isspace():
            cleaned_text += char
    return cleaned_text
def pertanyaan_lanjut(args):
    answer = input("Lanjut bertanya?\n")
    if answer.lower() in ["y","ya"]:
        main(args)
    if answer.lower() in ["t","tidak"]:
        exit()
    print("Jawaban yang diberikan salah")
    pertanyaan_lanjut(args)
def main(args):
    os.system("clear")
    print("Menggunakan", args.file)
    pertanyaan = str(input("Apa Pertanyaan yang diajukan?\n"))
    file = open(args.file,"r")
    text = file.read()
    teks_yang_dibersihkan = clean_text(text)
    print("sedang mencari jawabannya!")
    inputs = tokenizer.encode_plus(pertanyaan, teks_yang_dibersihkan, return_tensors="pt", max_length=1280, truncation=True)    
    outputs = model(**inputs)
    posisi_jawaban_awal = outputs.start_logits.argmax(dim=-1).item()
    posisi_jawaban_akhir = outputs.end_logits.argmax(dim=-1).item()
    jawaban = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][posisi_jawaban_awal:posisi_jawaban_akhir+1]))
    os.system("clear")
    print("Pertanyaannya:", pertanyaan)
    print("Jawabanya:", jawaban)
    pertanyaan_lanjut(args)
    

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file',
        type=str, 
        default='./The Google story.txt',
        help="Untuk mendefinisikan alamat file sumber jawaban",
    )
    args = parser.parse_args()
    main(args)