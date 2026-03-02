import json
import time
from openai import OpenAI
from typing import List, Dict, Optional


def llm_as_judge(predictions: List[Dict],
                 model: str = "gpt-4o",
                 openai_api_key: Optional[str] = None,
                 delay: float = 1.) -> List[Dict]:
    """
    Uses an LLM to judge the quality of zero shot text classification predictions.

    Args:
        predictions: Output of model.forward_predict – list of dicts with keys "text" and "scores".
        model: OpenAI model identifier (default "gpt-4o").
        openai_api_key: API key. If None, reads from environment variable OPENAI_API_KEY.
        delay: Seconds to wait between API calls to avoid rate limits.

    Returns:
        List of dicts with keys:
            - "text": original text
            - "scores": original scores
            - "llm_score": integer rating (1–5)
            - "justification": string explanation from the LLM
            - "error": (optional) if the API call failed
    """
    # initialize openAI client
    if openai_api_key:
        client = OpenAI(openai_api_key=openai_api_key)
    else:
        client = OpenAI()

    # prompt template
    prompt_template = """You are an expert judge of text classification systems.
    Given a text and a set of labels with predicted relevance
    scores (0–1, higher means more relevant), evaluate how well the scores reflect
    the true relevance of each label to the text.

    Text: "{text}"
    Predicted scores: {score_dict}

    Respong with:
    1. A single overall score from 1 to 5 (1 = completely wrong predictions, 5 = perfect).
    2. A brief justification (one or two sentences) explaining your rating.
    Return your answer as a JSON object with keys "score" (integer)
    and "justification" (string).
    """

    results = []

    for item in predictions:
        text = item["text"]
        scores = item["scores"]  # dict label -> score

        # Format the score dictionary as a string for the prompt
        score_str = ", ".join([f'"{k}": {v:.2f}' for k, v in scores.items()])

        prompt = prompt_template.format(text=text, score_dict=score_str)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0,
                response_format={"type": "json_object"})
            content = response.choices[0].message.content
            judge_result = json.loads(content)
            llm_score = judge_result.get("score")
            justification = judge_result.get("justification", "")

            # Validate score is integer between 1 and 5
            if not isinstance(llm_score,
                              int) or llm_score < 1 or llm_score > 5:
                llm_score = None
                justification = "Invalid score format from LLM."

        except Exception as e:
            llm_score = None
            justification = f"API error: {str(e)}"

        # Append result
        results.append({
            "text": text,
            "scores": scores,
            "llm_score": llm_score,
            "justification": justification
        })

        # Small delay to avoid hitting rate limits
        time.sleep(delay)

    return results


# Example usage
if __name__ == "__main__":
    # Example model output
    example_predictions = [{
        'text': 'I love machine learning.',
        'scores': {
            'AI': 0.97,
            'Finance': 0.39,
            'Politics': 0.05
        }
    }, {
        'text': 'Deep learning models are powerful.',
        'scores': {
            'Deep Learning': 0.98,
            'Neural Networks': 0.5,
            'Agriculture': 0.9
        }
    }]

    # API key set as environment variable OPENAI_API_KEY
    # import os; os.environ["OPENAI_API_KEY"] = "abc"
    judgments = llm_as_judge(example_predictions, model="gpt-4o")
    for j in judgments:
        print(f"Text: {j['text']}")
        print(f"Scores: {j['scores']}")
        print(f"LLM Score: {j['llm_score']}")
        print(f"Justification: {j['justification']}\n")
