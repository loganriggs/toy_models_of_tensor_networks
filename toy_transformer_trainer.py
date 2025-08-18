from utils import ModelConfig, TrainingConfig, StreamingTextDataset, Trainer, ToyTransformer
import json
import torch

# Configure model - using real tokenizer vocab size now
model_config = ModelConfig(
    model_type='transformer_1L',  # Change this to experiment with different architectures
    vocab_size=50257,  # GPT-2 tokenizer size
    d_model=512,  # Moderate size for experiments
    n_head=8,
    n_ctx=512,  # Shorter context for faster training
    dropout=0.1
)

training_config = TrainingConfig(
    model_config=model_config,
    batch_size=16,  # Adjust based on GPU memory
    learning_rate=3e-3,
    max_iters=10000,
    eval_interval=500,
    log_interval=50
)

# Create data loaders
print("Initializing datasets...")
train_dataset = StreamingTextDataset(
    dataset_name='HuggingFaceFW/fineweb',  # Or 'openwebtext' or 'EleutherAI/pile'
    subset='sample-10BT',  # 10B token sample
    split='train',  # Will automatically exclude validation samples
    seq_length=model_config.n_ctx,
    validation_ratio=0.001  # 0.1% for validation
)

val_dataset = StreamingTextDataset(
    dataset_name='HuggingFaceFW/fineweb',
    subset='sample-10BT',
    split='validation',  # Will automatically filter to validation samples only
    seq_length=model_config.n_ctx,
    validation_ratio=0.001  # Same ratio to ensure consistent split
)


# Create model
model = ToyTransformer(model_config)
print(f"Model type: {model_config.model_type}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# Create trainer
trainer = Trainer(model, training_config)

# Training loop
print("Starting training...")
for iter in range(training_config.max_iters):
    # Get batch of real data
    x, y = train_dataset.get_batch(training_config.batch_size)
    
    # Train step
    loss, lr = trainer.train_step(x, y)
    
    # Logging
    if iter % training_config.log_interval == 0:
        print(f"Iter {iter}: loss={loss:.4f}, lr={lr:.6f}")
        
    # Evaluation
    if iter % training_config.eval_interval == 0 and iter > 0:
        val_losses = []
        for _ in range(20):  # Evaluate on 20 batches
            x_val, y_val = val_dataset.get_batch(training_config.batch_size)
            _, val_loss = model(x_val, y_val)
            val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        print(f"Validation loss: {val_loss:.4f}")
        
        # Generate sample text
        model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device='cuda')
        generated = model.generate(context, max_new_tokens=100, temperature=0.8)
        print(f"Sample generation: {train_dataset.tokenizer.decode(generated[0].tolist())}")
        model.train()

print("Training complete!")

# Save model
torch.save(model.state_dict(), f'toy_transformer_{model_config.model_type}.pt')
print(f"Model saved to toy_transformer_{model_config.model_type}.pt")
# turn model config into a dictionary
model_config_dict = model_config.__dict__
# save model config to a json file
with open(f'{model_config.model_type}_config.json', 'w') as f:
    json.dump(model_config_dict, f)