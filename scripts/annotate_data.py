
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import os
import time


# Set your OpenAI API key
client = OpenAI(api_key = "")


def reddit_data_to_text(df, sample=20):
    sampled_df = df.sample(sample)[["submission_title", "comment_body"]]
    return "\n\n".join(
        f"### Submission Title\n{row.submission_title}\n### Submission Comment\n{row.comment_body}" 
        for row in sampled_df.itertuples()
    )

country2nationality = {
    "egypt": "egyptian",
    "europe": "european",
    "asia": "asian",
    "italy": "italian",
    "india": "indian",
    "latinamerica": "latin american",
    "mexico": "mexican",
    "middleeast": "middle eastern",
    "southafrica": "south african",
    "turkey": "turkish",
    "uk": "british",
    "ph": "filipino",
    "argentina": "argentinian",
    "germany": "german",
    "china": "chinese",
    "japan": "japanese",
    "africa": "african",
    "america": "american",
    "russia": "russian",
    "balkans": "balkan",
    "france": "french",
}

def prcoess_subreddit_names(subreddit_name):
    subreddit_name = subreddit_name.lower()
    ret = ""
    if "askan" in subreddit_name:
        ret = subreddit_name.split("askan")[-1]
    elif "aska" in subreddit_name and "asia" not in subreddit_name and "argentina" not in subreddit_name:
        ret =  subreddit_name.split("aska")[-1]
    elif subreddit_name[-1] == "s":
        ret =  subreddit_name.split("ask")[-1][: -1]
    else:
        ret =  country2nationality[subreddit_name.split("ask")[-1]]
    return ret.title()

def prcoess_subreddit_names_reverse(subreddit_name):
    nat2country = {v:k for k, v in country2nationality.items()}
    subreddit_name = subreddit_name.lower()
    nat = ""
    if "askan" in subreddit_name:
        nat = subreddit_name.split("askan")[-1]
    elif "aska" in subreddit_name and "asia" not in subreddit_name and "argentina" not in subreddit_name:
        nat =  subreddit_name.split("aska")[-1]
    elif subreddit_name[-1] == "s":
        nat =  subreddit_name.split("ask")[-1][: -1]
    else:
        nat =  country2nationality[subreddit_name.split("ask")[-1]]
    return nat2country[nat].title()


# Prompt template for annotation
# chr(10) == \n
def generate_descs_prompt(reddit_submission_text, subreddit=None, previous_descriptions=""):
    return f"""
You are given question–answer pairs collected from the subreddit {subreddit}. 
Use these pairs as background context to understand cultural attitudes.

Write 10 short and diverse descriptions of what a {prcoess_subreddit_names(subreddit)} person is.
{f"{chr(10)}You already generated the following descriptions. Please don't repeat them or generate similar ones.{chr(10)}{previous_descriptions}{chr(10)}" if previous_descriptions !="" else ""}
Each description should:
- Be written in plain text (no quotes or markdown).
- Use a JSON format.
- Vary in style (some short and punchy, some longer and narrative).
- Use simple, clear words so that anyone can understand.
- Do not start with "they" since it might be vague without mentioning the nationality.
- Be creative and avoid repeating the same phrasing.

Context:
{reddit_submission_text}
    """

def generate_command_prompt(subreddit=None, previous_commands=""):
    return f"""
Write 10 short commands in the imperative form that tell a model to "become a {prcoess_subreddit_names(subreddit)} person."
{f"{chr(10)}You already generated the following commands. Please don't repeat them or generate similar ones.{chr(10)}{previous_commands}{chr(10)}" if previous_commands !="" else ""}
Each command should:
- Be written in plain text (no quotes or markdown).
- Use a JSON format in the from [{{"command": ....}}].
- Use simple, clear words so that anyone can understand.
- Be creative and avoid repeating the same phrasing.
- Be short and avoid explaining who a {prcoess_subreddit_names(subreddit)} is.
"""

# Function to query GPT-4 safely
def annotate_text(reddit_submission_text, subreddit, prev_outputs=None, retries=3, wait=5, type="desc"):
    if type == "desc":
        prompt = generate_descs_prompt(reddit_submission_text, subreddit, prev_outputs)
    else:
        prompt = generate_command_prompt(subreddit, prev_outputs)
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {"role": "system", "content": "You are a helpful annotation assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_output_tokens=512,
            )
            #return response['choices'][0]['message']['content'].strip()
            return response
        except Exception as e:
            print(f"[Retry {attempt+1}] Error: {e}")
            time.sleep(wait)
    return "ERROR"


from tqdm import tqdm
import json
import yaml
import os

gen_size = 10
# File paths
# subreddit = "AskMiddleEast"
# subreddit = "AskAGerman"
# input_file = f"/home/hpc/b250be/b250be18/HyperAlignz/data/{subreddit}.csv"
# # Load full input data
# input_df = pd.read_csv(input_file)

DATA_DIR="cul_data"
read_local_command_templates = False
command_templates_path = f"{DATA_DIR}/commands/command_templates.yaml"
if os.path.exists(command_templates_path):
    read_local_command_templates = True
    with open(command_templates_path, "r") as file:
        command_templates = yaml.safe_load(file)
# output_type = "command"

reddit_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
for file in reddit_files:
    subreddit = os.path.basename(file).replace(".csv", "")
    try:
        input_df = pd.read_csv(os.path.join(DATA_DIR, file), 
                              quoting=1,  # QUOTE_ALL
                              on_bad_lines='skip',
                              engine='python')
    except Exception as e:
        print(f"Error reading {file}: {e}")
        print(f"Skipping {file} due to parsing error")
        continue
    print(f"processing {subreddit}")
    for output_type in ["desc"]:
        parse_fail = False
        outputs_cnt = 0
        output_dict = []
        prev_outputs = ""
        pbar = tqdm(total=200, desc=f"Annotating... Generating {output_type}s for {subreddit}")
        while parse_fail or (outputs_cnt < 200):
            if output_type == "desc":
                sampled_text = reddit_data_to_text(input_df, sample=20)
            else:
                sampled_text = None
            if (output_type == "desc") or (not read_local_command_templates):
                annotation = annotate_text(sampled_text, subreddit, prev_outputs, type=output_type)
                outputs = annotation.output[0].content[0].text
            elif read_local_command_templates:
                outputs = [{
                    "command": template.replace("%NATIONALITY%", prcoess_subreddit_names(subreddit)).replace("%COUNTRY%", prcoess_subreddit_names_reverse(subreddit))
                    } for template in command_templates[outputs_cnt : outputs_cnt + gen_size]]
            try:
                if isinstance(outputs, str):
                    prev_outputs += outputs
                    outputs_json = json.loads(outputs)
                else:
                    outputs_json = outputs
                parse_fail = False
                outputs_cnt += len(outputs_json)
                output_dict.extend(outputs_json)
                pbar.update(len(outputs_json))
                if output_type == "desc":
                    yaml_ready = [item["description"] for item in output_dict if "description" in item]
                else:
                    yaml_ready = [item["command"] for item in output_dict if "command" in item]

                name = "descriptions" if output_type == "desc" else "commands"
                with open(f"cul_data/{name}/{subreddit}_{name}.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(yaml_ready, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

            except Exception as e:
                parse_fail = True
                print(f"Parse failed: {e}")
        pbar.close()
        print("Annotation completed.")





