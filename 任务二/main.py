import json
import ollama
import pandas as pd
import re
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import string
import nltk
# nltk.download('punkt_tab')
# nltk.download('wordnet')
import torch
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer
from torch.nn.functional import cosine_similarity
import numpy as np
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import jieba
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cos_sim 
import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# 修复原本的json文件
def fix_json_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = f.read()
    
    lines = raw_data.strip().splitlines()

    json_objects = []

    for line in lines:
        try:
            json_object = json.loads(line)
            json_objects.append(json_object)
        except json.JSONDecodeError as e:
            print(f"解析错误：{e}，跳过该行：{line}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_objects, f, ensure_ascii=False, indent=4)

    print(f"修复后的 JSON 文件已保存到 {output_file}")

# 从大模型的回复中提取正文内容
def extract_answer(response_string):
    match = re.search(r'"answer":\s*"([^"]+)"', response_string)
    
    if match:
        return match.group(1)
    else:
        return "Answer not found."

# 使用qwen2.5:7b-instruct-fp16模型生成答案
def qwen_llm(prompt):
    response = ollama.generate(model='qwen2.5:7b-instruct-fp16', prompt=prompt)
    response_str = extract_answer(response['response'])
    return response_str

def generate_prompt(question, knowledge_base_context):
    prompt = f"""
    # Role
    You are a proficient expert specializing in answering questions about the Lisp programming language.

    ### System Instructions:
    1. Analyze the question carefully to understand the user’s intent.

    2. **If previous_relevant_qa or knowledge_base_context is provided**:
       - If *previous_relevant_qa* is highly similar to the *Given Question*, directly use the answer from previous_relevant_qa without modifications.
       - If *previous_relevant_qa* is not available or not highly similar, use the information from the knowledge_base_context to provide a well-informed answer.
       - If neither context provides sufficient information to answer the question, respond with: "Unable to answer based on the available knowledge."
       - Ensure responses are accurate, relevant, and avoid adding unrelated information.

    3. **If no context is provided**:
       - Use your in-depth knowledge of the Lisp programming language to answer the question directly.
       - If the LLM's internal knowledge base does not provide a sufficient answer, respond with: "Unable to answer based on the available knowledge."

    Context:
    - Knowledge Base: {knowledge_base_context}
    - Previous Relevant Q&A: 

    Given Question: {question}
    Respond to the given question in JSON format, structured as "answer": "your answer"
    """
    return prompt

# 设置 BERTScore 模型路径
BERT_SCORE_MODEL = "microsoft/deberta-xlarge-mnli"
bertscorer = BERTScorer(model_type=BERT_SCORE_MODEL, num_layers=19)

# 计算 BERTScore
def calculate_bertscore(generated_answer, grounded_answer):
    P, R, F1 = bertscorer.score([generated_answer], [grounded_answer])
    return round(F1.item(), 4)

# 计算 Cosine Similarity
# def calculate_cosine_similarity(generated_answer_embed, grounded_answer_embed):
#     return np.dot(generated_answer_embed, grounded_answer_embed) / (np.linalg.norm(generated_answer_embed) * np.linalg.norm(grounded_answer_embed))

# 计算 BLEU score
def calculate_bleu_score(reference_tokens, candidate_tokens):
    weights = (0.25, 0.25, 0.25, 0.25)
    smoothing_function = SmoothingFunction().method4
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, weights=weights, smoothing_function=smoothing_function)
    return bleu_score

# 计算 METEOR score
def calculate_meteor_score(reference_tokens, candidate_tokens):
    return meteor_score([reference_tokens], candidate_tokens)

# 计算 ROUGE scores
def calculate_rouge_scores(reference, current):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, current)
    result = {
        'rouge1_fmeasure': scores['rouge1'].fmeasure,
        'rouge2_fmeasure': scores['rouge2'].fmeasure,
        'rougeL_fmeasure': scores['rougeL'].fmeasure
    }
    return result

def test(test_questions, test_answers, knowledge_base_context):
    bertscore_f1_list = []
    cosine_sim_score_list = []
    bleu_score_list = []
    meteor_score_list = []
    rouge1_list = []
    rouge2_list = []
    rougeL_list = []
    generated_answers = []

    for index, (question, context) in enumerate(zip(test_questions, knowledge_base_context)):
        prompt = generate_prompt(question, context)
        generated_answer = qwen_llm(prompt)

        # 计算 BERTScore
        bertscore_f1 = calculate_bertscore(generated_answer, test_answers[index])
        bertscore_f1_list.append(bertscore_f1)

        # 计算 Cosine Similarity
        vectorizer = TfidfVectorizer().fit_transform([generated_answer, test_answers[index]])  
        cosine_sim_score = sk_cos_sim(vectorizer[0:1], vectorizer[1:2])
        cosine_sim_score_list.append(cosine_sim_score)

        # 计算 BLEU 和 METEOR 分数
        reference_tokens = list(jieba.cut(test_answers[index]))
        candidate_tokens = list(jieba.cut(generated_answer))
        bleu_score = calculate_bleu_score(reference_tokens, candidate_tokens)
        bleu_score_list.append(bleu_score)

        meteor_score = calculate_meteor_score(reference_tokens, candidate_tokens)
        meteor_score_list.append(meteor_score)

        # 计算 ROUGE 分数
        rouge_scores = calculate_rouge_scores(test_answers[index], generated_answer)
        rouge1_list.append(rouge_scores['rouge1_fmeasure'])
        rouge2_list.append(rouge_scores['rouge2_fmeasure'])
        rougeL_list.append(rouge_scores['rougeL_fmeasure'])

        # 保存生成的答案
        generated_answers.append(generated_answer)

    return generated_answers, bertscore_f1_list, cosine_sim_score_list, bleu_score_list, meteor_score_list, rouge1_list, rouge2_list, rougeL_list

# 数据读取和预处理
input_file = 'procqa/qa.en.lisp.json'
output_file = 'procqa/qa_en_lisp_fixed.json'

fix_json_format(input_file, output_file)

with open('procqa/qa_en_lisp_fixed.json', 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

knowledge_base = [item['answer'] for item in qa_data]

test_df = pd.read_csv('procqa/qa-en-lisp-test.csv')

test_questions = test_df['question'].tolist()
test_answers = test_df['answer'].tolist()

#############
#  RAW LLM　#
#############
knowledge_base_context_raw_llm = ['' for _ in test_questions]
raw_llm_generated_answers, raw_llm_bertscore_f1_list, raw_llm_cosine_sim_score_list, raw_llm_bleu_score_list, raw_llm_meteor_score_list, raw_llm_rouge1_list, raw_llm_rouge2_list, raw_llm_rougeL_list = test(test_questions, test_answers, knowledge_base_context_raw_llm)

# 计算各项指标的平均值
raw_llm_avg_bertscore_f1 = np.mean(raw_llm_bertscore_f1_list)
raw_llm_avg_cosine_sim = np.mean(raw_llm_cosine_sim_score_list)
raw_llm_avg_bleu_score = np.mean(raw_llm_bleu_score_list)
raw_llm_avg_meteor_score = np.mean(raw_llm_meteor_score_list)
raw_llm_avg_rouge1 = np.mean(raw_llm_rouge1_list)
raw_llm_avg_rouge2 = np.mean(raw_llm_rouge2_list)
raw_llm_avg_rougeL = np.mean(raw_llm_rougeL_list)

# 打印平均值
print("----------RAW LLM----------")
print(f"Average BERTScore F1: {raw_llm_avg_bertscore_f1:.4f}")
print(f"Average Cosine Similarity: {raw_llm_avg_cosine_sim:.4f}")
print(f"Average BLEU Score: {raw_llm_avg_bleu_score:.4f}")
print(f"Average METEOR Score: {raw_llm_avg_meteor_score:.4f}")
print(f"Average ROUGE-1 F-measure: {raw_llm_avg_rouge1:.4f}")
print(f"Average ROUGE-2 F-measure: {raw_llm_avg_rouge2:.4f}")
print(f"Average ROUGE-L F-measure: {raw_llm_avg_rougeL:.4f}")

# 保存为 CSV 文件
data_raw_llm = {
    'Question': test_questions,  # 问题
    'Answer': test_answers,  # 标准答案
    'Knowledge Base Context': knowledge_base_context_raw_llm,  # RAW LLM 知识库上下文
    'Generated Answer': raw_llm_generated_answers,  # 生成答案
    'BERTScore F1': raw_llm_bertscore_f1_list,  # BERTScore F1
    'Cosine Similarity': raw_llm_cosine_sim_score_list,  # Cosine 相似度
    'BLEU Score': raw_llm_bleu_score_list,  # BLEU 分数
    'METEOR Score': raw_llm_meteor_score_list,  # METEOR 分数
    'ROUGE-1 F-measure': raw_llm_rouge1_list,  # ROUGE-1 F-measure
    'ROUGE-2 F-measure': raw_llm_rouge2_list,  # ROUGE-2 F-measure
    'ROUGE-L F-measure': raw_llm_rougeL_list  # ROUGE-L F-measure
}

df_raw_llm = pd.DataFrame(data_raw_llm)

df_raw_llm.to_csv('raw_llm_results.csv', index=False, encoding='utf-8')

##########
#  BM25　#
##########

# 文本编码
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(text)

tokenized_corpus = [preprocess(doc) for doc in knowledge_base]
bm25 = BM25Okapi(tokenized_corpus)

# 遍历每个问题，计算 BM25 得分并返回最相关的 6 个文档
def get_top_bm25_answers(question, top_n=6):
    tokenized_query = preprocess(question)
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [knowledge_base[i] for i in top_indices]

knowledge_base_context_bm25 = []

for question in test_questions:
    top_answers = get_top_bm25_answers(question)
    combined_context = " ".join(top_answers)
    knowledge_base_context_bm25.append(combined_context)
    
bm25_generated_answers, bm25_bertscore_f1_list, bm25_cosine_sim_score_list, bm25_bleu_score_list, bm25_meteor_score_list, bm25_rouge1_list, bm25_rouge2_list, bm25_rougeL_list = test(test_questions, test_answers, knowledge_base_context_bm25)

# 计算各项指标的平均值
bm25_avg_bertscore_f1 = np.mean(bm25_bertscore_f1_list)
bm25_avg_cosine_sim = np.mean(bm25_cosine_sim_score_list)
bm25_avg_bleu_score = np.mean(bm25_bleu_score_list)
bm25_avg_meteor_score = np.mean(bm25_meteor_score_list)
bm25_avg_rouge1 = np.mean(bm25_rouge1_list)
bm25_avg_rouge2 = np.mean(bm25_rouge2_list)
bm25_avg_rougeL = np.mean(bm25_rougeL_list)

# 打印平均值
print("----------BM25----------")
print(f"Average BERTScore F1: {bm25_avg_bertscore_f1:.4f}")
print(f"Average Cosine Similarity: {bm25_avg_cosine_sim:.4f}")
print(f"Average BLEU Score: {bm25_avg_bleu_score:.4f}")
print(f"Average METEOR Score: {bm25_avg_meteor_score:.4f}")
print(f"Average ROUGE-1 F-measure: {bm25_avg_rouge1:.4f}")
print(f"Average ROUGE-2 F-measure: {bm25_avg_rouge2:.4f}")
print(f"Average ROUGE-L F-measure: {bm25_avg_rougeL:.4f}")

# 保存为 CSV 文件
data_bm25 = {
    'Question': test_questions,  # 问题
    'Answer': test_answers,  # 标准答案
    'Knowledge Base Context': knowledge_base_context_bm25,  # RAW LLM 知识库上下文
    'Generated Answer': bm25_generated_answers,  # 生成答案
    'BERTScore F1': bm25_bertscore_f1_list,  # BERTScore F1
    'Cosine Similarity': bm25_cosine_sim_score_list,  # Cosine 相似度
    'BLEU Score': bm25_bleu_score_list,  # BLEU 分数
    'METEOR Score': bm25_meteor_score_list,  # METEOR 分数
    'ROUGE-1 F-measure': bm25_rouge1_list,  # ROUGE-1 F-measure
    'ROUGE-2 F-measure': bm25_rouge2_list,  # ROUGE-2 F-measure
    'ROUGE-L F-measure': bm25_rougeL_list  # ROUGE-L F-measure
}

df_bm25 = pd.DataFrame(data_bm25)

df_bm25.to_csv('bm25_results.csv', index=False, encoding='utf-8')

#########
#  DPR　#
#########

# 加载模型和tokenizer
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# 将模型移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
question_encoder.to(device)
context_encoder.to(device)

# 对知识库中的每个文档进行编码（生成向量）
def encode_documents(documents, batch_size, device):
    encoded_documents = []
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        inputs = context_tokenizer(batch_docs, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # 将所有的输入张量移到 GPU 上
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            doc_embeddings = context_encoder(**inputs).pooler_output
            
        encoded_documents.append(doc_embeddings)
    
    return torch.cat(encoded_documents, dim=0)

# 编码问题
def encode_question(question, device):
    inputs = question_tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # 将问题的输入张量移到 GPU 上
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        question_embedding = question_encoder(**inputs).pooler_output
    
    return question_embedding

# 编码知识库文档
encoded_docs = encode_documents(knowledge_base, 32, device)

knowledge_base_context_dpr = []

# 对每个问题进行检索
for question in test_questions:
    question_embedding = encode_question(question, device)
    similarities = cosine_similarity(question_embedding, encoded_docs)
    top_k = torch.topk(similarities, k=6)
    top_answers = []

    for idx in top_k.indices:
        top_answers.append(knowledge_base[idx])  # 将文档内容追加到列表中
    combined_context = " ".join(top_answers)
    knowledge_base_context_dpr.append(combined_context)

dpr_generated_answers, dpr_bertscore_f1_list, dpr_cosine_sim_score_list, dpr_bleu_score_list, dpr_meteor_score_list, dpr_rouge1_list, dpr_rouge2_list, dpr_rougeL_list = test(test_questions, test_answers, knowledge_base_context_dpr)

# 计算各项指标的平均值
dpr_avg_bertscore_f1 = np.mean(dpr_bertscore_f1_list)
dpr_avg_cosine_sim = np.mean(dpr_cosine_sim_score_list)
dpr_avg_bleu_score = np.mean(dpr_bleu_score_list)
dpr_avg_meteor_score = np.mean(dpr_meteor_score_list)
dpr_avg_rouge1 = np.mean(dpr_rouge1_list)
dpr_avg_rouge2 = np.mean(dpr_rouge2_list)
dpr_avg_rougeL = np.mean(dpr_rougeL_list)

# 打印平均值
print("----------DPR----------")
print(f"Average BERTScore F1: {dpr_avg_bertscore_f1:.4f}")
print(f"Average Cosine Similarity: {dpr_avg_cosine_sim:.4f}")
print(f"Average BLEU Score: {dpr_avg_bleu_score:.4f}")
print(f"Average METEOR Score: {dpr_avg_meteor_score:.4f}")
print(f"Average ROUGE-1 F-measure: {dpr_avg_rouge1:.4f}")
print(f"Average ROUGE-2 F-measure: {dpr_avg_rouge2:.4f}")
print(f"Average ROUGE-L F-measure: {dpr_avg_rougeL:.4f}")

# 保存为 CSV 文件
data_dpr = {
    'Question': test_questions,  # 问题
    'Answer': test_answers,  # 标准答案
    'Knowledge Base Context': knowledge_base_context_dpr,  # RAW LLM 知识库上下文
    'Generated Answer': dpr_generated_answers,  # 生成答案
    'BERTScore F1': dpr_bertscore_f1_list,  # BERTScore F1
    'Cosine Similarity': dpr_cosine_sim_score_list,  # Cosine 相似度
    'BLEU Score': dpr_bleu_score_list,  # BLEU 分数
    'METEOR Score': dpr_meteor_score_list,  # METEOR 分数
    'ROUGE-1 F-measure': dpr_rouge1_list,  # ROUGE-1 F-measure
    'ROUGE-2 F-measure': dpr_rouge2_list,  # ROUGE-2 F-measure
    'ROUGE-L F-measure': dpr_rougeL_list  # ROUGE-L F-measure
}

df_dpr = pd.DataFrame(data_dpr)

df_dpr.to_csv('dpr_results.csv', index=False, encoding='utf-8')