import json

# Pre-filled values from your training command
config = {
    'model': {
        'positional_encoding': 't5_relative',
        'model_size': 'small',
        # We'll ask for vocab_size, max_seq_len, etc.
    },
    'training': {
        'batch_size': 2,
        'epochs': 5,
        'use_wandb': True,
        # We'll ask for learning_rate, tokenizer_name, etc.
    }
}

def ask(prompt, default=None, type_cast=str):
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    val = input(prompt)
    if val.strip() == '' and default is not None:
        return default
    return type_cast(val)

# Model config
config['model']['vocab_size'] = ask('Vocab size (len(tokenizer))', 50257, int)
config['model']['max_seq_len'] = ask('Max sequence length', 512, int)

# Training config
config['training']['learning_rate'] = ask('Learning rate', 1e-4, float)
config['training']['tokenizer_name'] = ask('Tokenizer name', 'gpt2', str)

# Save config
out_path = ask('Output config JSON file', 'config_for_eval.json', str)
with open(out_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"Config saved to {out_path}") 