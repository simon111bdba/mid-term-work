from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModelForCausalLM.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("./gpt2_model")

prompt = "Once upon a time, in a land far far away,"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

num_words = 200  

# 计算max_length，假设每个单词平均对应1个token
# 需要减去提示文本中已有的token数量
prompt_length = len(tokenizer.encode(prompt))
real_max_length = num_words + prompt_length

# 生成文本
output = model.generate(input_ids, max_length = real_max_length, num_return_sequences=1, temperature=1.7, repetition_penalty=1.7)

# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
