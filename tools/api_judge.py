from tqdm import tqdm
import multiprocessing
import json
import re
from openai import OpenAI
from src.api import generate_text_chat
from argparse import ArgumentParser

client = OpenAI()


general_checklist = """
- Instruction adherence is a core evaluation metric. If the AI's response does not fully follow the user's instructions, it constitutes a serious error. 
- Correctness is a core evaluation metric. If the AI's response contains factual inaccuracies, it would be a serious flaw.
- If the AI's response contain a large amount of repetitive content, it would be a serious flaw.
"""

meta_prompt = """
# Instructions

You are an evaluation expert. Your task is to assess the quality of AI model responses. We will provide you with user queries and AI responses. Please note that both the user queries and AI responses are in audio format. For your convenience, we have converted them into text, but you should evaluate from the perspective of voice communication and analyze the characteristics of voice communication when assessing the quality of the AI response.
You should first carefully read the user query to analyze the task, then evaluate the quality of the response based on the rules provided below.

# Conversation between User and AI

### User Query
<|begin_of_query|>

{query}

<|end_of_query|>

### AI Response
<|begin_of_response|>

{response}

<|end_of_response|>

# Evaluation

## Checklist
<|begin_of_checklist|>

{checklist}

<|end_of_checklist|>

The checklist serves as a guiding framework for your evaluation. However, feel free to consider aspects beyond its contents to ensure a well - rounded assessment.

## Rules

You should evaluate based on the analysis of user questions and AI responses, referring to the contents in the checklist during the evaluation. However, remember that the checklist is meant to provide comprehensive reference information, but it is not the standard answer. Sometimes, the AI response does not need to cover all the contents involved in the checklist to meet user needs, and you need to make this judgment on your own. The scoring scale ranges from 1 to 10:
- 1~2 points: No value/meaningless. The AI response contains many factual errors or serious flaws, or is irrelevant to the user query, providing little to no value to the user.
- 3~4 points: Partially valuable/meaningful. The AI response contains several factual errors or serious flaws, or poorly meets the user's requirements, but has some redeeming qualities and offers partial value to the user.
- 5~6 points: Flawed. The AI response has some issues, such as minor factual errors/flaws, or does not fully meet the user's requirements. However, these are relatively minor, and the response generally satisfies the user's needs.
- 7~8 points: Meets requirements. The AI response satisfies the user's needs well, with no major flaws or errors, or only very minor issues that do not affect overall quality.
- 9~10 points: High quality. The AI response perfectly meets the user's requirements, with virtually no room for improvement.

## Output Format
First, analyze the query itself and understand the user's intent. Then provide your analysis of the model's response. Summarize your evaluation in two aspects: "Strengths" and "Weaknesses". Finally, write your score. The score should appear on the last line in the following format:
Score: [your score]
"""


def extract_rating(llm_output):
    pattern = r"Score: (?:\[)?(\d+)(?:\])?"
    match = re.search(pattern, llm_output)
    if match:
        return int(match.group(1))
    else:
        return None

def generate(item):
    query = item['user_query']
    responses = item['response']
    results = []
    for response in responses:
        query_checklist = item['checklist']
        checklist = general_checklist + query_checklist
        prompt = meta_prompt.format(query=query, response=response, checklist=checklist)
        rtn = generate_text_chat(
            client=client,
            model='gpt-4o-mini',
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=0.5, top_p=0.95, n=1
        ).choices[0].message.content.strip()
        results.append(rtn)
    return results

def main():
    parser = ArgumentParser()
    parser.add_argument('--src_file', required=True)
    parser.add_argument('--tgt_file', required=True)
    args = parser.parse_args()

    with open(args.src_file, 'r') as f:
        length = sum(1 for _ in f)
    
    # Then open the file again to process it
    with open(args.src_file, 'r') as f, open(args.tgt_file, 'w') as fout:
        for line in tqdm(f, total=length):
            json_obj = json.loads(line.strip())
            evaluation_raw = generate(json_obj)
            json_obj['evaluation_raw'] = evaluation_raw
            scores = []
            for evaluation in evaluation_raw:
                score = extract_rating(evaluation)
                scores.append(score)
            json_obj['score'] = scores
            fout.write(json.dumps(json_obj) + "\n")

if __name__ == '__main__':
    main()
    # text = "The user query is objective as it seeks specific information regarding which OECD countries have the highest physician salaries.\n\n### Analysis of the AI Response\n\nThe AI response fails to adequately address the user query, which specifically asks for a list of OECD countries with the highest physician salaries. Instead of providing this information, the AI diverts the conversation to a more casual tone, discussing friendship and general chat, which is irrelevant to the user's request.\n\n**Advantages:**\n- The AI acknowledges that physician salaries vary based on specialties, experience, and location, which is a relevant point in the context of the query.\n- The suggestion to check specific sources for up-to-date information is a reasonable approach, as salary data can change frequently.\n\n**Disadvantages:**\n- The response does not provide any specific countries or data regarding physician salaries, which is the core of the user's question.\n- The tone of the response is overly casual and does not align with the serious nature of the inquiry about physician salaries.\n- There is a lack of concrete information or statistics that would have been beneficial for the user, which makes the response largely uninformative.\n- The response contains irrelevant content that does not pertain to the user's request, which detracts from its overall quality.\n\nGiven these points, the AI response does not meet the user's requirements and fails to provide valuable information.\n\nScore: 2"
    # print(extract_rating(text))