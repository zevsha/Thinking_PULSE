import html
import re
import numpy as np

def normalize(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)     # normalize whitespace
    return text

def get_answers_predictions(file_path):
    answers = []
    llm_predictions = []
    current_answer = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Answer:"):
                current_answer = line.replace('Answer:', '').strip()[1:-1]
            elif line.startswith("LLM:") and current_answer is not None:
                llm_prediction = line.replace('LLM:', '').strip()
                answers.append(current_answer)
                llm_predictions.append(llm_prediction)
                current_answer = None  # Reset for next pair

    return answers, llm_predictions

             

def evaluate(answers, llm_predictions, k=1):
    NDCG = 0.0
    HT = 0.0
    predict_num = min(len(answers), len(llm_predictions))  # Ensure equal length
    print("Total examples:", predict_num)

    for i in range(predict_num):
        answer = normalize(answers[i])
        prediction = normalize(llm_predictions[i])
        #answer = answers[i]
        #prediction = llm_predictions[i]

        if k == 1:
            if answer in prediction:
                NDCG += 1
                HT += 1
        else:
            try:
                rank = prediction.index(answer)
                if rank < k:
                    NDCG += 1 / np.log2(rank + 2)
                    HT += 1
            except:
                pass

    return NDCG / predict_num, HT / predict_num


if __name__ == "__main__":
    inferenced_file_path = 'results file path'
    answers, llm_predictions = get_answers_predictions(inferenced_file_path)
    print(len(answers), len(llm_predictions))
    #assert(len(answers) == len(llm_predictions))
    
    ndcg, ht = evaluate(answers, llm_predictions, k=1)
    print(f"ndcg at 1: {ndcg}")
    print(f"hit at 1: {ht}")