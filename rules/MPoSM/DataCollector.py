import torch
import logging
from transformers import DataCollatorForLanguageModeling

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomDataCollatorForPOS(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15, mask_on="detailed"):
        logging.info(f"Initializing Data Collator with mlm_probability: {mlm_probability}, mask_on: {mask_on}")
        super().__init__(tokenizer, mlm, mlm_probability)
        self.mask_on = mask_on

    def mask_tokens(self, inputs, labels):
        logging.info(f"Masking tokens with mode: {self.mask_on}")
        if self.mask_on == "random":
            self.mask_on = random.choice(["general", "detailed"])
        
        if self.mask_on == "general":
            masked_inputs, masked_labels = self._mask_on_sequence(inputs['general'], labels['general'])
        else:
            masked_inputs, masked_labels = self._mask_on_sequence(inputs['detailed'], labels['detailed'])
        
        logging.info(f"Masked {sum(masked_inputs)} tokens.")
        return masked_inputs, masked_labels

    def _mask_on_sequence(self, input_sequence, label_sequence):
        logging.info("Masking sequence...")
        labels = input_sequence.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Masking with handling punctuation
        for i, token in enumerate(input_sequence):
            if "_" in token:  # Handle tokens with punctuation
                word_part, punct_part = token.split("_")
                if masked_indices[i]:  
                    input_sequence[i] = f"[MASK]_{punct_part}"

        labels[~masked_indices] = -100  # Only compute loss for masked tokens
        inputs = input_sequence.clone()
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        logging.info(f"Masked sequence successfully.")
        return inputs, labels
