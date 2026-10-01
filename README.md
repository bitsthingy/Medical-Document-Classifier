# Medical Document Classifier

Classifying clinical transcriptions into their **medical specialty** by fine-tuning transformer models.

## Main Idea

Given the raw text of a medical transcription, predict which specialty it belongs to (Surgery, Radiology, Neurology, etc.). The project compares a domain-specific model (**Bio_ClinicalBERT**) against a long-context model (**Longformer**), and tests how much the label set itself affects performance by removing document-type labels that are not true specialties.

## Dataset

- [Medical Transcriptions (MTSamples)](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) on Kaggle
- 4,999 transcriptions across 40 specialties
- Top 10 classes kept, duplicate transcriptions removed (1,440 dropped)
- Only the `transcription` text is used as input (no keywords, sample name or description)

## Approach

| Step | Details |
|------|---------|
| Cleaning | Dropped nulls, stripped labels, kept top 10 specialties, deduplicated |
| Splitting | Stratified 5-fold split with 10% of train held out for validation (fold 0 used for results) |
| Long text handling | Head + tail truncation to fit the token limit (512 for ClinicalBERT, 1024 for Longformer) |
| Imbalance | Class-weighted loss |
| Training | Hugging Face `Trainer`, early stopping on validation macro-F1 |
| Models | `emilyalsentzer/Bio_ClinicalBERT`, `allenai/longformer-base-4096` |
| Experiments | (A) 10 classes, (B) 8 classes after removing document-type labels (Consult - History and Phy., SOAP / Chart / Progress Notes) |

## Results (fold 0)

| Setup | Model | Accuracy | Macro-F1 |
|-------|-------|----------|----------|
| A: 10 classes | Bio_ClinicalBERT | 0.793 | 0.688 |
| B: 8 classes | Bio_ClinicalBERT | 0.870 | **0.772** |
| B: 8 classes | Longformer | **0.873** | 0.726 |

- Removing document-type labels gave the biggest gain, since "Consult" and "SOAP" notes overlap in content with real specialties.
- Longformer has slightly higher accuracy, but Bio_ClinicalBERT has better macro-F1 because it handles the smaller classes better.
- Surgery is classified very well (F1 about 0.97); Neurology and Gastroenterology remain the hardest.

## Run It

```bash
pip install transformers accelerate pandas scikit-learn torch matplotlib seaborn kaggle
```

Open `Medical_Document_Classifier.ipynb` (built for Google Colab with a GPU). It needs a `kaggle.json` API key to download the dataset.

## Tech Stack

Python · PyTorch · Hugging Face Transformers · scikit-learn · pandas · matplotlib · seaborn
