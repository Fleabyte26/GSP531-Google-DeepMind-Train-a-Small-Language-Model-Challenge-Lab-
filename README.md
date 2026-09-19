# GSP531-Google-DeepMind-Train-a-Small-Language-Model-Challenge-Lab-
Google DeepMind: Train a Small Language Model (Challenge Lab)

Task 1: Setup & Import
In the Google Cloud Console, navigate to Agent Platform > Notebooks > Colab Enterprise > My notebooks.

Select your assigned region and click Import.

Choose Cloud Storage, provide the notebook path given on the lab instructions page, and click Import.

Once gdm_challenge_lab.ipynb opens, click the top right runtime expander, choose Connect to an existing runtime, and connect to colab-cpu-runtime.

Run the initial cells under Task 1 to import libraries and load the Arabic stories dataset.

Task 2: Configure the Character Tokenizer
In the SimpleArabicCharacterTokenizer class cell, complete the two methods:


class SimpleArabicCharacterTokenizer:
    def __init__(self):
        # Existing initialization logic provided in the notebook
        pass

    def character_tokenize(self, text: str) -> list[str]:
        """Converts raw Arabic text into a list of single-character tokens."""
        # [TODO - Add your code here]
        return list(text)

    def join_text(self, tokens: list[str]) -> str:
        """Joins a list of character tokens back into a single string without padding."""
        # [TODO - Add your code here]
        return "".join(tokens)


Run the test cell below the class to ensure it passes, then click Check my progress for Task 2.

Task 3: Generate Text from an N-gram Model
In the generate_text_from_ngram_model function, implement the sampling switch (greedy vs. random sampling from the transition probabilities/counts) and return the joined tokens:


def generate_text_from_ngram_model(model, prompt, max_tokens=100, sampling_mode="greedy"):
    """
    Generates text from an n-gram model using greedy or random sampling.
    """
    n = model.n if hasattr(model, 'n') else getattr(model, 'order', 3)
    generated_tokens = list(prompt)

    for _ in range(max_tokens):
        # Extract the current context (last n-1 tokens)
        context = tuple(generated_tokens[-(n - 1):]) if n > 1 else ()

        # Retrieve next token distribution / candidates from the model
        candidates = model.get_candidates(context) if hasattr(model, 'get_candidates') else model.get(context, {})

        if not candidates:
            break

        tokens = list(candidates.keys())
        probabilities = list(candidates.values())

        # [TODO - Add your code here: Support either random or greedy sampling]
        if sampling_mode == "greedy":
            # Select the token with the highest probability/count
            next_token = tokens[np.argmax(probabilities)]
        elif sampling_mode == "random":
            # Normalize probabilities if necessary and sample
            probs = np.array(probabilities, dtype=np.float64)
            probs = probs / probs.sum()
            next_token = np.random.choice(tokens, p=probs)
        else:
            raise ValueError(f"Unsupported sampling mode: {sampling_mode}")

        generated_tokens.append(next_token)

    # Return generated tokens as a single joined string
    return "".join(generated_tokens)


    (Note: Check the variable names inside your notebook's stub for generate_text_from_ngram_model. If the notebook already computes probabilities / candidates or provides a helper, adapt the variable names accordingly while keeping np.argmax(probs) for greedy and np.random.choice(..., p=...) for random).

Run the test cell to verify, then click Check my progress for Task 3.

Task 4: Prepare the Dataset for Training
Part 1: Build segment_encoded_sequence
This function breaks a sequence of token IDs into overlapping subsequences of up to max_length. For language model inputs/targets, standard non-overlapping or stride-based chunking is used:


def segment_encoded_sequence(encoded_sequence: list[int], max_length: int) -> list[list[int]]:
    """
    Segments an encoded sequence of token IDs into subsequences of length max_length.
    The final subsequence can be shorter.
    """
    # [TODO - Add your code here]
    subsequences = []
    for i in range(0, len(encoded_sequence), max_length):
        subsequences.append(encoded_sequence[i:i + max_length])
    return subsequences


    Part 2: Training Sequences & Shifting (create_training_sequences)


    def create_training_sequences(dataset, tokenizer, max_length: int, pad_token_id: int = 0):
    """
    Encodes, segments, pads sequences, and splits into input (X) and target (y) arrays.
    """
    all_subsequences = []

    for text in dataset:
        encoded = tokenizer.encode(text) if hasattr(tokenizer, 'encode') else [tokenizer.vocab[c] for c in text if c in tokenizer.vocab]
        subsequences = segment_encoded_sequence(encoded, max_length)
        all_subsequences.extend(subsequences)

    inputs = []
    targets = []

    for seq in all_subsequences:
        if len(seq) < 2:
            continue

        # Shift sequences: input gets seq[:-1], target gets seq[1:]
        inp = seq[:-1]
        tar = seq[1:]

        # Pad remaining length to (max_length - 1)
        pad_len = (max_length - 1) - len(inp)
        if pad_len > 0:
            inp = inp + [pad_token_id] * pad_len
            tar = tar + [pad_token_id] * pad_len

        inputs.append(inp)
        targets.append(tar)

    return np.array(inputs), np.array(targets)


    Verification: Run the final test blocks to produce the arrays and verify all green checkmarks on the lab assessment page.

💡 Troubleshooting & Common Pitfalls
Runtime Disconnects: Colab Enterprise sessions can time out if idle. Check the top-right indicator to verify colab-cpu-runtime is active before executing code cells.

Array Shapes: Ensure targets and inputs match (N, max_length - 1) after accounting for next-token shift offsets.

Pad Tokens: Ensure the default padding ID matches the notebook specification (0 or tokenizer-defined <pad>).
