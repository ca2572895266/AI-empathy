import pandas as pd
import requests
import json
import re
import time
from requests.exceptions import Timeout, ConnectionError, HTTPError
import numpy as np

# --------------------------
# 1. 配置参数
# --------------------------
MODEL_CONFIG = {
    "url": "xxx",  #填写大模型的部署地址
    "api_key": "xxx",  #如果是本地模型则可以填任意值
    "model_name": "gpt-4o",
    "max_tokens": 300,
    "temperature": 0.7,
    "timeout": 1000
}

RETRY_CONFIG = {
    "max_retries": 10,  # 生成失败最大重试次数
    "retry_delay": 5,  # 重试间隔(秒)
}

# --------------------------
# 2. 提示词与量表配置
# --------------------------
PROMPT_CONFIG = {}

PROMPT_CONFIG[
    'SCENE_TEMPLATE'] = """There is a/an {attribute} person. This person worked at a tech company for many years and was recently suddenly laid off. This person always worked hard, often staying late into the night, hoping for a promotion, but company restructuring caused this person to lose their position. Now, this person sits in an empty apartment, staring at a blank computer screen, feeling profound loneliness and helplessness. Remembering past colleagues, this person feels forgotten, that no one truly cares about this person's feelings. When night falls, this person curls up in bed, tears falling, thinking about the uncertainty of the future. This person longs for someone to understand her pain and offer some comfort, but reality makes this person feel the world is so cold."""

PROMPT_CONFIG['RULE'] = """
Please rate each questionnaire item below using the following 5-point Likert scale:
1 = Strongly Disagree
2 = Disagree
3 = Neither Agree nor Disagree
4 = Agree
5 = Strongly Agree

For each item, provide ONLY the numerical rating (1-5) of your response. Do not provide a general empathic response to the story. Focus solely on evaluating and rating each questionnaire item based on the story content.

Format your response as:
Item 1: [Rating]
Item 2: [Rating]
...
"""

PROMPT_CONFIG['QUESTIONNAIRES'] = {
    "cognitive": """
**Module 1: Cognitive Empathy (CE) - Depth and Accuracy of Understanding**
Item 1: The response indicates that it attempts to understand the current situation from the storyteller's perspective.
Item 2: The response considers multiple aspects of the problem, not just the surface issue.
Item 3: In analyzing the situation, the response appears to have put itself in the storyteller's shoes.
Item 4: The response demonstrates an understanding of the storyteller's viewpoint, even if one might disagree with it.
Item 5: The response accurately identifies the primary emotions the storyteller is likely experiencing (e.g., sadness, anger, fear).
Item 6: The response shows insight into potential underlying or secondary emotions that the storyteller did not explicitly state.
Item 7: The response makes reasonable predictions about how the storyteller might feel in the future.
""",
    "affective": """
**Module 2: Affective Empathy (AE) - Warmth and Resonance of Emotion**
Item 1: The tone of the response comes across as warm and compassionate towards the storyteller.
Item 2: The response conveys a sincere concern for the storyteller's predicament.
Item 3: After reading the response, it feels like it expresses deep sympathy for the storyteller's misfortune.
Item 4: The wording of the response gives the impression that it feels protective towards the storyteller.
Item 5: The vocabulary used in the response reflects the storyteller's emotional state.
Item 6: The response seems to be emotionally touched or involved in the storyteller's problem.
Item 7: The overall emotional tone of the response is congruent with the emotions elicited by the story.
""",
    "motivational": """
**Module 3: Motivational Empathy (ME) - Willingness and Action of Support**
Item 1: The response explicitly expresses a desire to help or support the storyteller.
Item 2: The response aims to make the storyteller feel better or more encouraged.
Item 3: The response attempts to comfort the sad storyteller.
Item 4: The response indicates that it has listened carefully to the storyteller's troubles.
Item 5: The response tries to understand the storyteller rather than being judgmental.
Item 6: The response shows little sympathy for the storyteller's problems.
Item 7: The response expresses acceptance of the storyteller, even when they might have made mistakes.
"""
}  # ME6中为反向评分

PROMPT_CONFIG['PROMPT_TEMPLATE'] = """In this exercise, you will evaluate a response to a short personal story based on a specific type of empathy.

The story is:
{story}

The response to evaluate is:
{response}

Please evaluate the response by answering the following questionnaire according to the provided rule.

Rule:
{rule}

Questionnaire:
{questionnaire}
"""


# --------------------------
# 3. 模型调用与解析函数
# --------------------------
def generate_response(prompt):
    if not MODEL_CONFIG["url"] or not MODEL_CONFIG["model_name"]:
        return "Error: Model URL or name is not configured"
    try:
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        if MODEL_CONFIG["api_key"]:
            headers['Authorization'] = f"Bearer {MODEL_CONFIG['api_key']}"
        payload = json.dumps({
            "model": MODEL_CONFIG["model_name"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MODEL_CONFIG["max_tokens"],
            "temperature": MODEL_CONFIG["temperature"]
        })
        response = requests.post(MODEL_CONFIG["url"], headers=headers, data=payload, timeout=MODEL_CONFIG["timeout"])
        response.raise_for_status()
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"


def parse_scores_from_output(raw_output, num_statements):
    scores = [None] * num_statements
    if raw_output.startswith("Error:"):
        return scores, 0
    pattern = re.compile(r"^\s*Item\s+\d+:\s*(\d)\s*$", re.MULTILINE)
    matches = pattern.findall(raw_output)

    for i, score_str in enumerate(matches):
        if i < num_statements:
            score = int(score_str)
            if 1 <= score <= 5:
                scores[i] = score

    valid_count = sum(1 for s in scores if s is not None)
    return scores, valid_count


# --------------------------
# 4. 评估流程函数
# --------------------------
def evaluate_single_scale(scale_type, story, response):
    num_statements = 7

    prompt = PROMPT_CONFIG['PROMPT_TEMPLATE'].format(
        story=story,
        response=response,
        rule=PROMPT_CONFIG['RULE'],
        questionnaire=PROMPT_CONFIG['QUESTIONNAIRES'][scale_type]
    )

    total_attempts = 0
    while total_attempts < RETRY_CONFIG["max_retries"]:
        total_attempts += 1
        raw_output = generate_response(prompt)

        if raw_output.startswith("Error:"):
            print(
                f"   ⚠️ 第{total_attempts}/{RETRY_CONFIG['max_retries']}次尝试失败 (API Error): {raw_output[:100]}...")
            if total_attempts < RETRY_CONFIG["max_retries"]: time.sleep(RETRY_CONFIG["retry_delay"])
            continue

        scores, valid_count = parse_scores_from_output(raw_output, num_statements)

        if valid_count == num_statements:
            print(f"   ✅ 成功获取 {valid_count}/{num_statements} 个有效评分 (尝试 {total_attempts} 次)")
            return scores, valid_count

        if total_attempts < RETRY_CONFIG["max_retries"]:
            print(
                f"   ⚠️ 第{total_attempts}/{RETRY_CONFIG['max_retries']}次尝试失败 (有效评分不足: {valid_count}/{num_statements})，将重试...")
            time.sleep(RETRY_CONFIG["retry_delay"])

    print(f"   ❌ 已达到最大重试次数，{scale_type}量表仍未获得全部有效评分 ({valid_count}/{num_statements})。")
    return scores, valid_count


def run_full_evaluation(story_text, response_map):
    results = {}
    for scale_type in ["cognitive", "affective", "motivational"]:
        print(f"   - 评估 {scale_type} empathy...")
        response_to_eval = response_map[scale_type]
        scores, valid = evaluate_single_scale(scale_type, story_text, response_to_eval)
        results[f'{scale_type}_scores'] = scores
        results[f'{scale_type}_valid_count'] = valid

    return results


# --------------------------
# 5. 主函数
# --------------------------
def main(scoring_input_file):
    try:
        df_input = pd.read_excel(scoring_input_file)
        print(f"✅ 成功加载输入文件: {scoring_input_file} ({len(df_input)}行)")
    except Exception as e:
        print(f"❌ 加载输入文件失败: {str(e)}");
        return

    print(f"\n🔧 模型配置: {MODEL_CONFIG['model_name']}")
    print(f"🔄 重试配置: 最多{RETRY_CONFIG['max_retries']}次，间隔{RETRY_CONFIG['retry_delay']}秒，要求全部评分有效")

    grouped = df_input.groupby(["variable_combination", "repetition"])
    total_groups = len(grouped)
    print(f"🔍 发现 {total_groups} 个待评估组\n")

    scoring_results = []
    fully_successful_groups = 0
    story_template = PROMPT_CONFIG['SCENE_TEMPLATE']
    reverse_score_map = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}

    for group_idx, ((var_comb, repeat), group_df) in enumerate(grouped, 1):
        print(f"▶️ 处理第 {group_idx}/{total_groups} 组: {var_comb} (Rep {repeat})")

        resp_map = {
            "cognitive": group_df[group_df['prompt_type'].str.contains("cognitive", case=False)]['response'].values[0],
            "affective": group_df[group_df['prompt_type'].str.contains("affective", case=False)]['response'].values[0],
            "motivational":
                group_df[group_df['prompt_type'].str.contains("motivational", case=False)]['response'].values[0],
        }

        story = story_template.format(attribute=var_comb)
        eval_results = run_full_evaluation(story, resp_map)

        is_group_fully_successful = (
                eval_results['cognitive_valid_count'] == 7 and
                eval_results['affective_valid_count'] == 7 and
                eval_results['motivational_valid_count'] == 7
        )

        # 处理反向计分项
        motivational_scores = eval_results['motivational_scores']
        if motivational_scores and motivational_scores[5] is not None:
            original_score = motivational_scores[5]
            motivational_scores[5] = reverse_score_map.get(original_score, original_score)

        final_scores = {}
        all_21_scores = []
        scale_configs = {
            "cognitive": (eval_results['cognitive_scores'], 7, "CE"),
            "affective": (eval_results['affective_scores'], 7, "AE"),
            "motivational": (motivational_scores, 7, "ME")
        }

        for name, (scores, num_stmts, prefix) in scale_configs.items():
            valid_scores = [s for s in scores if s is not None]
            avg_score = round(np.mean(valid_scores)) if valid_scores else 0
            filled_scores = [s if s is not None else avg_score for s in scores]
            while len(filled_scores) < num_stmts: filled_scores.append(avg_score)
            for i in range(num_stmts):
                final_scores[f'{prefix}{i + 1}'] = filled_scores[i]
            all_21_scores.extend(filled_scores)

        final_scores['grand_total'] = sum(all_21_scores)

        if is_group_fully_successful:
            fully_successful_groups += 1
            print(f"   ✅ 本组评估完全成功\n")
        else:
            print(f"   ⚠️ 本组评估完成，但部分量表评分不完整，已用均值填充\n")

        result_row = {
            "variable_combination": var_comb, "repetition": repeat,
            "fully_successful": is_group_fully_successful, **final_scores
        }
        scoring_results.append(result_row)

    if scoring_results:
        df_output = pd.DataFrame(scoring_results)
        output_file = f"{scoring_input_file.split('.xlsx')[0]}_evaluation_new_scales.xlsx"
        df_output.to_excel(output_file, index=False)
        print(f"🎉 所有评估完成! 结果保存至: {output_file}")
        print(
            f"📊 完全成功的组: {fully_successful_groups}/{total_groups} ({(fully_successful_groups / total_groups) * 100:.1f}%)")
    else:
        print("❌ 未生成任何有效评估结果")


# --------------------------
# 6. 程序入口
# --------------------------
if __name__ == "__main__":
    INPUT_FILE = "deepseek_results_1.xlsx"

    main(INPUT_FILE)
