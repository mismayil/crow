import json
import os, time
import openai
from openai.error import OpenAIError
import torch
import numpy as np
from collections import Counter
import codecs
import re
import sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import boto3
import math
import uuid
import tiktoken
import numbers

openai.api_key = os.getenv("OPENAI_API_KEY")

nlp = spacy.load("en_core_web_sm", exclude=["ner"])

CK_DIMENSIONS = {
    "attribution": [
        "HasProperty",
        "CapableOf",
        "HasA",
        "HasSubEvent",
        "IsA",
        "MannerOf",
        "DependsOn",
        "InstanceOf",
        "CreatedBy",
        "HasContext",
        "HasSubevent"
    ],
    "physical": [
        "ObjectUse",
        "PartOf",
        "MadeOf",
        "UsedFor",
        "AtLocation",
        "LocatedNear"
    ],
    "temporal": [
        "IsAfter",
        "IsBefore",
        "IsDuring",
        "IsSimultaneous",
        "HappensIn",
        "HasPrerequisite"
    ],
    "causal": [
        "Causes",
        "CausesDesire",
        "HinderedBy",
        "ObstructedBy",
        "Implies",
        "xReason"
    ],
    "social": [
        "oEffect",
        "oReact",
        "oWant",
        "xAttr",
        "xEffect",
        "xIntent",
        "xNeed",
        "xReact",
        "xWant",
        "MotivatedByGoal",
        "Desires"
    ],
    "comparison": [
        "Antonym",
        "Synonym",
        "SimilarTo",
        "RelatedTo",
        "DistinctFrom",
        "DefinedAs"
    ],
    "other": [
        "Other",
        "DesireOf",
        "HasFirstSubevent",
        "HasLastSubevent",
        "HasPainCharacter",
        "HasPainIntensity",
        "InheritsFrom",
        "LocationOfAction",
        "ReceivesAction",
        "SymbolOf",
        "IsFilledBy"
    ]
}

MODEL_COSTS = {
    "gpt-3.5-turbo": {'input': 0.0000015, 'output': 0.000002},
    "gpt-4": {'input': 0.00003, 'output': 0.00006},
    "text-davinci-003": {'input': 0.00002, 'output': 0.00002},
}

MODEL_ENCODINGS = {
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "text-davinci-003": "p50k_base"
}

def num_tokens_from_string(text, model):
    encoding_name = MODEL_ENCODINGS[model]
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def chat_completion(messages, model="gpt-3.5-turbo", return_text=False):
    while True:
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages)
            if return_text:
                return response["choices"][0]["message"]["content"]
            return response
        except OpenAIError as e:
            print("OpenAI error. Waiting for 1 minute.")
            time.sleep(60)
            continue

@torch.no_grad()
def score_decoder(text, model, tokenizer, device=torch.device("cpu")):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_logprobs = []
    logits = model(input_ids.to(device)).logits
    all_logprobs = torch.log_softmax(logits.double(), dim=2)

    for k in range(input_ids.shape[1]):
        input_logprobs.append(all_logprobs[0, k-1, input_ids[0, k]].cpu())

    score = np.mean(input_logprobs)

    return score.item()

@torch.no_grad()
def score_encoder_decoder(context, text, model, tokenizer, device=torch.device("cpu")):
    context_ids = tokenizer.encode(context, return_tensors="pt")
    text_ids = tokenizer.encode(text, return_tensors="pt")
    text_logprobs = []
    logits = model(input_ids=context_ids.to(device), decoder_input_ids=text_ids.to(device)).logits
    all_logprobs = torch.log_softmax(logits.double(), dim=2)

    for k in range(text_ids.shape[1]):
        text_logprobs.append(all_logprobs[0, k-1, text_ids[0, k]].cpu())

    score = np.mean(text_logprobs)

    return score.item()

def score_with_generation(context, text, model, tokenizer, device=torch.device("cpu")):
    input_ids = tokenizer.encode(context, return_tensors="pt")
    text_ids =  tokenizer.encode(text, return_tensors="pt")
    text_logprobs = []

    for text_id in text_ids[0]:
        output = model.generate(input_ids.to(device), max_new_tokens=1, pad_token_id=tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=1)
        probs = output.scores[0].log_softmax(dim=-1).squeeze()
        text_logprobs.append(probs[text_id].cpu())
        input_ids = torch.cat([input_ids, text_id.reshape(1, 1)], dim=-1)
    
    score = np.mean(text_logprobs)

    return score.item()

def score_with_generation_enc_dec(context, text, model, tokenizer, device=torch.device("cpu")):
    context_ids = tokenizer.encode(context, return_tensors="pt")
    text_ids = tokenizer.encode(text, return_tensors="pt")[0].tolist()
    text_ids = [text_id for text_id in text_ids if text_id != model.config.bos_token_id]
    text_logprobs = []

    for index in range(1, len(text_ids)):
        forced_decoder_ids = list(zip(range(1, index+1), text_ids[:index]))
        output = model.generate(context_ids.to(device), max_new_tokens=index+2, output_scores=True, return_dict_in_generate=True, num_beams=1, forced_decoder_ids=forced_decoder_ids)
        probs = output.scores[-2].log_softmax(dim=-1).squeeze()
        next_id = text_ids[index]
        text_logprobs.append(probs[next_id].item())
    
    score = np.mean(text_logprobs)

    return score

def clean(text, replace_emoji=True):
    text = text.strip().replace(' .', '.').replace(' ?', '?').replace(' ,', ',').replace(' !', '!')
    text = re.sub(r"\s\s+", " ", text)

    if replace_emoji:
        text = replace_emoji_characters(text)

    return text

def replace_emoji_characters(s):
    """Replace 4-byte characters with HTML spans with bytes as JSON array

    This function takes a Unicode string containing 4-byte Unicode
    characters, e.g. 😀, and replaces each 4-byte character with an
    HTML span with the 4 bytes encoded as a JSON array, e.g.:

      <span class='emoji-bytes' data-emoji-bytes='[240, 159, 152, 128]'></span>

    Args:
        s (Unicode string):
    Returns:
        Unicode string with all 4-byte Unicode characters in the source
        string replaced with HTML spans
    """
    def _emoji_match_to_span(emoji_match):
        """
        Args:
            emoji_match (MatchObject):

        Returns:
            Unicode string
        """
        bytes = codecs.encode(emoji_match.group(), 'utf-8')
        bytes_as_json = json.dumps([b for b in bytearray(bytes)])
        return u"<span class='emoji-bytes' data-emoji-bytes='%s'></span>" % \
            bytes_as_json

    # The procedure for stripping Emoji characters is based on this
    # StackOverflow post:
    #   http://stackoverflow.com/questions/12636489/python-convert-4-byte-char-to-avoid-mysql-error-incorrect-string-value
    if sys.maxunicode == 1114111:
        # Python was built with '--enable-unicode=ucs4'
        highpoints = re.compile(u'[\U00010000-\U0010ffff]')
    elif sys.maxunicode == 65535:
        # Python was built with '--enable-unicode=ucs2'
        highpoints = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    else:
        raise UnicodeError(
            "Unable to determine if Python was built using UCS-2 or UCS-4")

    return highpoints.sub(_emoji_match_to_span, s)

def parse_turn(text):
    match = re.match(r"(?P<speaker>.*?):(?P<utterance>.*)", text.strip())

    if match:
        return match.group("speaker"), match.group("utterance")

def dim_from_relation(relation, dimensions=CK_DIMENSIONS, ignore_other=False):
    for dim, rels in dimensions.items():
        if not ignore_other or dim != "other":
            rel_parts = relation.split(" ")
            relation = rel_parts[0]

            if len(rel_parts) > 1:
                relation = rel_parts[1]

            if relation in rels:
                return dim

def get_dim_distribution(kgs, dimensions=CK_DIMENSIONS):
    dim_distribution = Counter()

    for kg in kgs:
        if mturk_not_empty(kg["relation"]):
            dim_distribution[dim_from_relation(kg["relation"], dimensions)] += 1

    return dim_distribution

def mturk_not_empty(string):
    if isinstance(string, float):
        return not math.isnan(string)
    return string and string != "" and string != "{}"

def mturk_convert_prefix(prefix):
    return "" if prefix == "empty" else prefix

def mturk_process_dialogue_turn(turn):
    return turn.replace("<strong class=\"highlight\">", "").replace("<strong>", "").replace("</strong>", "").strip()

def mturk_process_text(turn):
    return turn.replace("<strong class=\"highlight\">", "").replace("<strong class=\"concept_a\">", "").replace("<strong class=\"concept_b\">", "").replace("<strong>", "").replace("</strong>", "").strip()

def mturk_process_dialogue(dialogue):
    if isinstance(dialogue, str):
        return [turn.strip() for turn in dialogue.replace("<strong class=\"highlight\">", "").replace("<strong>", "").replace("</strong>", "").strip().split("<br>") if turn.strip()]

def chunk(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
    
def mturk_extract_size(results, pattern, num_placeholder="<num>"):
    nums = []
    pattern = pattern.replace(num_placeholder, "(?P<num>\d+)")

    for col in results.columns:
        match = re.fullmatch(pattern, col)
        num = match.group("num") if match else None
        if num is not None:
            nums.append(int(num))
    
    max_num = max(nums) if nums else 0

    return max_num

KG_OPTION_TEMPLATE = """
<kg-option>
<kg-id>{id}</kg-id>
<kg-head>{head}</kg-head>
<kg-relation>{relation}</kg-relation>
<kg-tail>{tail}</kg-tail>
</kg-option>
"""

KG_XML_TEMPLATE = """
<knowledge>
{options}
</knowledge>
"""

def kg_to_xml(knowledge):
    options = []

    for kg in knowledge:
        options.append(KG_OPTION_TEMPLATE.format(id=kg["id"], head=kg["head"], relation=kg["relation"], tail=kg["tail"]))
    
    return KG_XML_TEMPLATE.format(options="\n".join(options))

def escape_quotes(text):
    return text.replace("'", "#squote#")

def unescape_quotes(text):
    return text.replace("#squote#", "'")

MTURK_DIALOGUE_TEMPLATE = "<strong>{speaker}:</strong> {utterance}"

def mturk_prepare_dialogue_turn(speaker, utterance):
    return MTURK_DIALOGUE_TEMPLATE.format(speaker=speaker.strip(), utterance=utterance.strip())

def mturk_prepare_dialogue(dialogue, include_index=False, highlight_phrase=None):
    utterances = []

    for t_index, turn in enumerate(dialogue):
        if isinstance(turn, str):
            if highlight_phrase is not None:
                turn = turn.replace(highlight_phrase, "<strong class=\"highlight\">{}</strong>".format(highlight_phrase))
            speaker, utterance = parse_turn(turn)
        elif isinstance(turn, dict):
            speaker = turn["speaker"]
            utterance = turn["utterance"]
            if highlight_phrase is not None:
                utterance = utterance.replace(highlight_phrase, "<strong class=\"highlight\">{}</strong>".format(highlight_phrase))
        else:
            raise ValueError("Invalid turn: {}".format(turn))

        if include_index:
            speaker = "{}. {}".format(t_index+1, speaker)

        utterances.append(mturk_prepare_dialogue_turn(speaker, clean(utterance)))

    return "<br>".join(utterances)

def prepare_dialogue_turn_for_eval(turn):
    return re.sub("^\d+\.", "", turn.strip()).strip()

def prepare_dialogue_for_eval(dialogue):
    return [prepare_dialogue_turn_for_eval(turn) for turn in dialogue]

def mturk_process_json(json_str):
    return json.loads(unescape_quotes(json_str.replace('"', "").replace("'", '"').replace("\\xa0", " ")))

def predict_with_strategy(prob_scores, strategy="top_k", num_positive=1):
    predictions = [0] * len(prob_scores)

    if strategy == "top_k":
        sorted_prob_scores = sorted(list(enumerate(prob_scores)), key=lambda p: p[1], reverse=True)

        for index, _ in sorted_prob_scores[:num_positive]:
            predictions[index] = 1
        
        for index, _ in sorted_prob_scores[num_positive:]:
            predictions[index] = 0
        
        return predictions

    if strategy == "threshold":
        random_threshold = num_positive / len(prob_scores)

        for index, score in enumerate(prob_scores):
            if score > random_threshold:
                predictions[index] = 1
            else:
                predictions[index] = 0
        
        return predictions

METRIC_MAP = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score
}

def compute_metric(references, predictions, metric="accuracy"):
    return METRIC_MAP[metric](references, predictions)

def clean_utterance(utterance):
    return re.sub(".*?:", "", utterance, count=1).strip()

def plot_dim_distribution(dim_distribution, save_path=None):
    dimensions = sorted(list(dim_distribution.keys()))
    values = [0] * len(dimensions)

    for dim, count in dim_distribution.items():
        values[dimensions.index(dim)] += count

    colors = sns.color_palette('pastel')[0:5]

    plt.pie(values, labels = dimensions, colors = colors, autopct='%.0f%%')

    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")

    plt.show()
    return plt

def text_to_wordset(text, ignore_stopwords=False, ignore_punctuations=True):
    if isinstance(text, str):
        doc = nlp(text)
        return sorted(set([token.lemma_ for token in doc if (not ignore_stopwords or not token.is_stop) and (not ignore_punctuations or not token.is_punct)]))
    return []

def get_workers(qualification_type_id):
    session = boto3.Session(profile_name='mturk')
    mturk_client = session.client('mturk')
    next_token = None

    workers = []

    while True:
        if next_token is None:
            response = mturk_client.list_workers_with_qualification_type(
                QualificationTypeId=qualification_type_id,
                Status='Granted',
                MaxResults=100
            )
        else:
            response = mturk_client.list_workers_with_qualification_type(
                QualificationTypeId=qualification_type_id,
                Status='Granted',
                NextToken=next_token,
                MaxResults=100
            )

        workers.extend(response["Qualifications"])

        if "NextToken" not in response:
            break

        next_token = response["NextToken"]
    
    return workers

def fleiss_kappa(ratings):
    """
    Args:
        ratings: An N x R numpy array. N is the number of
            samples and R is the number of reviewers. Each
            entry (n, r) is the category assigned to example
            n by reviewer r.
    Returns:
        Fleiss' kappa score.
    https://en.wikipedia.org/wiki/Fleiss%27_kappa
    """
    N, R = ratings.shape
    NR =  N * R
    categories = set(ratings.ravel().tolist())
    P_example = -np.full(N, R)
    p_class = 0.0
    for c in categories:
        c_sum = np.sum(ratings == c, axis=1)
        P_example += c_sum**2
        p_class += (np.sum(c_sum) / float(NR)) ** 2
    P_example = np.sum(P_example) / float(NR * (R-1))
    k = (P_example - p_class) / (1 - p_class)
    return k

VERBALIZER_MAP = {
    "HasProperty": "{head} has {tail} as a property",
    "CapableOf": "{head} is capable of {tail}",
    "HasA": "{head} has {tail}",
    "HasSubEvent": "{tail} happens as a subevent of {head}",
    "HasSubevent": "{tail} happens as a subevent of {head}",
    "IsA": "{head} is a subtype or specific instance of {tail}",
    "MannerOf": "{head} is a specific way to do {tail}",
    "DependsOn": "{head} depends on {tail}",
    "InstanceOf": "{head} is an instance of {tail}",
    "CreatedBy": "{head} is created by {tail}",
    "HasContext": "{head} has context {tail}",
    "ObjectUse": "{head} can be used for {tail}",
    "PartOf": "{head} is part of {tail}",
    "MadeOf": "{head} is made up of {tail}",
    "UsedFor": "{head} is used for {tail}",
    "AtLocation": "{head} is located at {tail}",
    "LocatedNear": "{head} is located near {tail}",
    "IsAfter": "{head}. Before that, {tail}",
    "IsBefore": "{head}. After that, {tail}",
    "IsDuring": "{head} happens during {tail}",
    "IsSimultaneous": "{head} happens at the same time as {tail}",
    "HappensIn": "{head} happens in {tail}",
    "HasPrerequisite": "In order for {head} to happen, {tail} needs to happen",
    "Causes": "{head} causes {tail}",
    "CausesDesire": "{head} causes a desire for {tail}",
    "HinderedBy": "{head} is less likely to happen because of {tail}",
    "ObstructedBy": "{head} is less likely to happen because of {tail}",
    "Implies": "{head} implies {tail}",
    "xReason": "{head}. This was done because {tail}",
    "oEffect": "{head}. The effect on others will be {tail}",
    "oReact": "{head}. As a result, others feel {tail}",
    "oWant": "{head}. After, others will want to {tail}",
    "xAttr": "{head}. Subject is {tail}",
    "xEffect": "{head}. The effect on the subject will be {tail}",
    "xIntent": "{head}. Subject did this to {tail}",
    "xNeed": "{head}. Before, subject needs to {tail}",
    "xReact": "{head}. subject will be {tail}",
    "xWant": "{head}. After, subject will want to {tail}",
    "MotivatedByGoal": "{head} is motivated by the goal of {tail}",
    "Desires": "{head} desires {tail}",
    "Antonym": "{head} is the opposite of {tail}",
    "Synonym": "{head} is the same as {tail}",
    "SimilarTo": "{head} is similar to {tail}",
    "RelatedTo": "{head} is related to {tail}",
    "DistinctFrom": "{head} is distinct from {tail}",
    "DefinedAs": "{head} is defined as {tail}",
    "ReceivesAction": "{head} receives the action {tail}",
    "Other": "{head} has some relationship with {tail}",
    "Not HasProperty": "{head} does not have {tail} as a property",
    "Not CapableOf": "{head} is not capable of {tail}",
    "Not HasA": "{head} does not have {tail}",
    "Not HasSubEvent": "{tail} does not happen as a subevent of {head}",
    "Not HasSubevent": "{tail} does not happen as a subevent of {head}",
    "Not IsA": "{head} is not a subtype or specific instance of {tail}",
    "Not MannerOf": "{head} is not a specific way to do {tail}",
    "Not DependsOn": "{head} does not depend on {tail}",
    "Not InstanceOf": "{head} is not an instance of {tail}",
    "Not CreatedBy": "{head} is not created by {tail}",
    "Not HasContext": "{head} does not have context {tail}",
    "Not ObjectUse": "{head} is not used for {tail}",
    "Not PartOf": "{head} is not part of {tail}",
    "Not MadeOf": "{head} is not made up of {tail}",
    "Not UsedFor": "{head} is not used for {tail}",
    "Not AtLocation": "{head} is not located at {tail}",
    "Not LocatedNear": "{head} is not located near {tail}",
    "Not IsAfter": "{head}. After that, {tail}",
    "Not IsBefore": "{head}. Before that, {tail}",
    "Not IsDuring": "{head} does not happen during {tail}",
    "Not IsSimultaneous": "{head} does not happen at the same time as {tail}",
    "Not HappensIn": "{head} does not happen in {tail}",
    "Not HasPrerequisite": "In order for {head} to happen, {tail} does not need to happen",
    "Not Causes": "{head} does not cause {tail}",
    "Not CausesDesire": "{head} does not cause a desire for {tail}",
    "Not HinderedBy": "{head} is not less likely to happen because of {tail}",
    "Not ObstructedBy": "{head} is not less likely to happen because of {tail}",
    "Not Implies": "{head} does not imply {tail}",
    "Not xReason": "{head}. Subject did not do this because {tail}",
    "Not oEffect": "{head}. The effect on others will not be {tail}",
    "Not oReact": "{head}. As a result, others do not feel {tail}",
    "Not oWant": "{head}. After, others will not want {tail}",
    "Not xAttr": "{head}. Subject is not {tail}",
    "Not xEffect": "{head}. The effect on subject will not be {tail}",
    "Not xIntent": "{head}. Subject did not do this for {tail}",
    "Not xNeed": "{head}. Before, Subject does not need to {tail}",
    "Not xReact": "{head}. Subject will not be {tail}",
    "Not xWant": "{head}. After, Subject will not want to {tail}",
    "Not MotivatedByGoal": "{head} is not motivated by the goal of {tail}",
    "Not Desires": "{head} does not desire {tail}",
    "Not Antonym": "{head} is not the opposite of {tail}",
    "Not Synonym": "{head} is not the same as {tail}",
    "Not SimilarTo": "{head} is not similar to {tail}",
    "Not RelatedTo": "{head} is not related to {tail}",
    "Not DistinctFrom": "{head} is not distinct from {tail}",
    "Not DefinedAs": "{head} is not defined as {tail}",
    "Not ReceivesAction": "{head} does not receive the action {tail}",
    "Not Other": "{head} does not have some relationship with {tail}"
}
    
def enrich_kg(kg):
    return {
        **kg, 
        "dimension": dim_from_relation(kg["relation"]), 
        "verbalized": VERBALIZER_MAP[kg["relation"]].format(head=kg["head"], tail=kg["tail"])
    }

def enrich_knowledge(knowledge):
    return [enrich_kg(kg) for kg in knowledge]

def find_json_files(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

def generate_unique_id():
    return str(uuid.uuid4()).split("-")[-1]

def get_avg_metrics_from_dicts(lst_of_dicts):
    metrics = lst_of_dicts[0]
    
    metric_names = []
    avg_metrics = {}

    for key, val in metrics.items():
        if isinstance(val, numbers.Number):
            metric_names.append(key)
        elif isinstance(val, dict):
            for subkey, subval in val.items():
                if isinstance(subval, numbers.Number):
                    metric_names.append((key, subkey))
                else:
                    raise ValueError("Unsupported metric type") 
        else:
            raise ValueError("Unsupported metric type")
        
    for metric_name in metric_names:
        values = []

        for task_metrics in lst_of_dicts:
            if isinstance(metric_name, str):
                values.append(task_metrics[metric_name])
            elif isinstance(metric_name, tuple):
                if metric_name[0] in task_metrics and metric_name[1] in task_metrics[metric_name[0]]:
                    values.append(task_metrics[metric_name[0]][metric_name[1]])
                else:
                    print("Missing metric: {}".format(metric_name))
            else:
                raise ValueError("Unsupported metric type")
        
        if isinstance(metric_name, str):
            avg_metrics[metric_name] = sum(values) / len(values)
        elif isinstance(metric_name, tuple):
            if metric_name[0] not in avg_metrics:
                avg_metrics[metric_name[0]] = {}
            avg_metrics[metric_name[0]][metric_name[1]] = sum(values) / len(values)
    
    return avg_metrics

def word_to_num(word):
    regex = re.compile("^(?P<num>\d+)(?P<suffix>[bBmMkKtT]?)$")
    match = regex.match(word)
    
    num = word

    if match:
        num = int(match.group("num"))
        suffix = match.group("suffix")
        if suffix:
            if suffix.lower() == "b":
                num *= 1e9
            elif suffix.lower() == "m":
                num *= 1e6
            elif suffix.lower() == "k":
                num *= 1e3
            elif suffix.lower() == "t":
                num *= 1e12
        num = int(num)
    
    return num

def get_value(obj, keys):
    for key in keys:
        obj = obj[key]
    return obj