from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 加载预训练的GPT-2模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 准备训练数据
train_path = 'input.txt'  # 替换为你的训练数据路径
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_path,
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./gpt2_model",          # 输出目录
    overwrite_output_dir=True,          # 覆盖输出目录
    num_train_epochs=3,                # 训练轮数
    per_device_train_batch_size=4,     # 每个设备的训练批次大小
    save_steps=10_000,                 # 保存模型的步数
    save_total_limit=2,                # 最多保存的模型数量
    prediction_loss_only=True,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model("./gpt2_model")

# 使用训练好的模型生成文本
model = GPT2LMHeadModel.from_pretrained("./gpt2_model")
model.eval()

# 输入文本
input_text = "hello, nice to me you, my name is"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 生成文本
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
