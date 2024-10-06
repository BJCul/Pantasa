from transformers import RobertaTokenizer, RobertaForMaskedLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from transformers import DataCollatorForLanguageModeling
import sagemaker
import random
import torch

# Custom Data Collator for Masking General or Detailed POS tags
class CustomDataCollatorForPOS(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15, mask_on="detailed"):
        super().__init__(tokenizer, mlm, mlm_probability)
        self.mask_on = mask_on

    def mask_tokens(self, inputs, labels):
        if self.mask_on == "random":
            self.mask_on = random.choice(["general", "detailed"])

        if self.mask_on == "general":
            masked_inputs, masked_labels = self._mask_on_sequence(inputs['general'], labels['general'])
        else:
            masked_inputs, masked_labels = self._mask_on_sequence(inputs['detailed'], labels['detailed'])

        return masked_inputs, masked_labels

    def _mask_on_sequence(self, input_sequence, label_sequence):
        labels = input_sequence.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        inputs = input_sequence.clone()
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        return inputs, labels

# Prepare the dataset and training script for SageMaker
def prepare_dataset_and_trainer(train_file_s3, output_dir, mask_on):
    # Load the tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('jcblaise/roberta-tagalog-base')
    model = RobertaForMaskedLM.from_pretrained('jcblaise/roberta-tagalog-base')

    # Load the dataset from S3
    dataset = load_dataset('csv', data_files={'train': train_file_s3}, split='train')

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Custom Data Collator
    data_collator = CustomDataCollatorForPOS(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        mask_on=mask_on  # Could be "general", "detailed", or "random"
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        learning_rate=2e-5,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

# Main function
if __name__ == "__main__":
    # Retrieve train file from S3
    train_file_s3 = "s3://your-bucket-name/path-to-dataset/processed_tagalog_data.csv"
    
    # Define output directory for the model
    output_dir = "/opt/ml/model"
    
    # Set mask_on to "random" for randomly switching between general and detailed
    mask_on = "random"  # Could be "general", "detailed", or "random"
    
    # Prepare and train the model
    prepare_dataset_and_trainer(train_file_s3, output_dir, mask_on)
