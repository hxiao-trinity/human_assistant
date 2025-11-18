from transformers import TextStreamer
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from tqdm import tqdm
import torch

# Initialize metrics
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bleu = BLEU()

# Storage for results
results = {
    'exact_match': [],
    'rouge1': [],
    'rouge2': [],
    'rougeL': [],
    'bleu': [],
    'predictions': [],
    'references': []
}

# Loop through all data points
print(f"Evaluating {len(test_dataset['conversations'])} samples...\n")

for idx in tqdm(range(len(test_dataset['conversations']))):
    # Prepare messages
    messages = [
        {'role': 'system', 'content': test_dataset['conversations'][idx][0]['content']},
        {'role': 'user', 'content': test_dataset['conversations'][idx][1]['content']}
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ).removeprefix('<bos>')

    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **tokenizer(text, return_tensors="pt").to("cuda"),
            max_new_tokens=125,
            temperature=1,
            top_p=0.95,
            top_k=64,
        )

    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated part (remove the prompt)
    prompt_length = len(tokenizer.encode(text, add_special_tokens=False))
    generated_tokens = outputs[0][prompt_length:]
    prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Get reference answer
    reference = test_dataset['conversations'][idx][2]['content'].strip()

    # Store predictions and references
    results['predictions'].append(prediction)
    results['references'].append(reference)

    # Calculate Exact Match
    exact_match = 1 if prediction.lower() == reference.lower() else 0
    results['exact_match'].append(exact_match)

    # Calculate ROUGE scores
    rouge_scores = rouge.score(reference, prediction)
    results['rouge1'].append(rouge_scores['rouge1'].fmeasure)
    results['rouge2'].append(rouge_scores['rouge2'].fmeasure)
    results['rougeL'].append(rouge_scores['rougeL'].fmeasure)

    # Calculate BLEU score (needs list format)
    bleu_score = bleu.sentence_score(prediction, [reference]).score / 100.0
    results['bleu'].append(bleu_score)

# Calculate average metrics
print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"Total samples: {len(results['exact_match'])}")
print(f"\nExact Match: {sum(results['exact_match']) / len(results['exact_match']) * 100:.2f}%")
print(f"ROUGE-1: {sum(results['rouge1']) / len(results['rouge1']):.4f}")
print(f"ROUGE-2: {sum(results['rouge2']) / len(results['rouge2']):.4f}")
print(f"ROUGE-L: {sum(results['rougeL']) / len(results['rougeL']):.4f}")
print(f"BLEU: {sum(results['bleu']) / len(results['bleu']):.4f}")
print("=" * 50)

# Optional: Print some example predictions
print("\nSample Predictions (first 3):")
print("-" * 50)
for i in range(min(3, len(results['predictions']))):
    print(f"\nSample {i + 1}:")
    print(f"Reference:  {results['references'][i]}")
    print(f"Prediction: {results['predictions'][i]}")
    print(f"Exact Match: {results['exact_match'][i]}")
    print(f"ROUGE-L: {results['rougeL'][i]:.4f}")