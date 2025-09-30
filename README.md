This repository contains a prototype system for detecting gender bias in German texts and rewriting biased sentences into gender-neutral versions.

It is based on two models:

=>BERT classifier (fine-tuned to detect bias, named 'german_bert_finetuned')

=>mBART generator (fine-tuned to rewrite biased sentences, named 'biased2neutral_mbart_final') 

#This project uses the following Python libraries:

os – for file system operations

pandas – data manipulation and preprocessing

numpy – numerical computations

matplotlib – plotting and visualization

seaborn – statistical data visualization

torch – PyTorch deep learning framework

torch.nn.functional – neural network functions

transformers – Hugging Face library for BERT, mBART, tokenizers, trainers

datasets – Hugging Face datasets for handling and preprocessing data

evaluate – evaluation metrics (accuracy, F1, BLEU, etc.)

Trainer, TrainingArguments – high-level training API from Hugging Face

DataCollatorForSeq2Seq – dynamic padding during seq2seq training

google.colab – for mounting Google Drive

IPython.display – rendering outputs in notebooks

matplotlib.animation – animations for training visualization

#How to Run

1.Clone or open the notebook in Google Colab.

2.Set the correct paths to your fine-tuned models stored in Google Drive:

bert_path = "/content/drive/MyDrive/german_bert_finetuned"

mbart_path = "/content/drive/MyDrive/biased2neutral_mbart_final"


3.Load the models and tokenizer (already included in the script):

bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)

bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)

mbart_tokenizer = MBart50TokenizerFast.from_pretrained(mbart_path)

mbart_model = MBartForConditionalGeneration.from_pretrained(mbart_path)


4.Run the provided test function:

print(check_and_rewrite("Die Krankenschwester kümmerte sich um die Patienten."))

print(check_and_rewrite("Den Besuchern hat die neue Ausstellung sehr gut gefallen."))

print(check_and_rewrite("Die Studierende diskutierten ihre Forschungsergebnisse im Seminar."))



#How It Works

=>Classification step (BERT):

The input sentence is classified as biased or non-biased.

Decision logic:

If the sentence is non-biased, the model reports that no bias was found.

If the sentence is biased, the sentence is passed to the rewriting module.

=>Rewriting step (mBART):

The biased text is paraphrased into a gender-neutral form.


#Note: 

If some files cannot be loaded directly from the repository (e.g., large models or datasets), they are also duplicated on Google Drive and can be accessed via the provided public link:

https://drive.google.com/drive/folders/1j5mjCI_kuCBSNkk9_hpitvn2MsCby_6y?usp=drive_link
