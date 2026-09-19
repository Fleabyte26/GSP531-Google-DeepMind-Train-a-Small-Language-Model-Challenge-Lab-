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


def generate_text_from_ngram_model(
    start_prompt: str,
    n_tokens: int,
    ngram_model: dict[str, dict[str, float]],
    tokenizer: SimpleArabicCharacterTokenizer,
    sampling_mode: Literal["random", "greedy"] = "random"
) -> str:
    """Generate text based on a starting prompt using an ngram model.

    Args:
        start_prompt: The initial prompt to start the generation.
        n_tokens: The number of tokens to generate after the prompt.
        ngram_model: An ngram model mapping contexts of n-1 tokens to distributions
            over next token.
        tokenizer: The tokenizer to encode and decode text.
        sampling_mode: Whether to use random or greedy sampling. Supported
            options are "random" and "greedy".

    Returns:
        The generated text from the prompt.
    """
    # Tokenize the starting prompt.
    start_tokens = tokenizer.character_tokenize(start_prompt)
    generated_tokens = start_tokens + []

    # Determine context length (n - 1) from the keys of the model
    sample_context = next(iter(ngram_model.keys()))
    context_len = len(sample_context)

    for _ in range(n_tokens):
        # Extract the current context string from the end of generated tokens
        current_context = tokenizer.join_text(generated_tokens[-context_len:])

        # Look up next token distribution for the current context
        candidates = ngram_model.get(current_context, {})

        if not candidates:
            break

        tokens = list(candidates.keys())
        probabilities = list(candidates.values())

        if sampling_mode == "greedy":
            # Pick token with the highest probability
            next_token = tokens[np.argmax(probabilities)]
        elif sampling_mode == "random":
            # Sample using random.choices with probability weights
            next_token = random.choices(tokens, weights=probabilities, k=1)[0]
        else:
            raise ValueError(f"Unsupported sampling mode: {sampling_mode}")

        generated_tokens.append(next_token)

    # Convert tokens back to string
    generated_text = tokenizer.join_text(generated_tokens)
    return generated_text

    (Note: Check the variable names inside your notebook's stub for generate_text_from_ngram_model. If the notebook already computes probabilities / candidates or provides a helper, adapt the variable names accordingly while keeping np.argmax(probs) for greedy and np.random.choice(..., p=...) for random).


Replace the last cell with this complete:


# Instantiate tokenizer first so it exists when building the n-gram model
tokenizer = SimpleArabicCharacterTokenizer()

# Train n-gram model from dataset.
n = 4 # Size of n-grams.
ngram_model = build_ngram_model(dataset, n, tokenizer)

# Generate text.
start_prompt = "يوم واحد"
print(f"Start prompt is:\n\t{start_prompt}")
n_tokens = 15 # Specify the number of new tokens to generate.

generated_text = generate_text_from_ngram_model(
    start_prompt,
    n_tokens,
    ngram_model,
    tokenizer,
    sampling_mode="random")

print(f"Text generated is:\n\t{display_arabic(generated_text)}")

# Do not remove or modify this logging call, it will be used for tracking purposes
logger.info(f'Task 3: The total word count for the generated text is: {len(generated_text.split())}')




Run the test cell to verify, then click Check my progress for Task 3.

Task 4: Prepare the Dataset for Training
Part 1: Build segment_encoded_sequence
This function breaks a sequence of token IDs into overlapping subsequences of up to max_length. For language model inputs/targets, standard non-overlapping or stride-based chunking is used:


def segment_encoded_sequence(
    encoded_sequence: list[int], 
    max_length: int, 
    overlap: int = 0
) -> list[list[int]]:
    """Segments an encoded sequence of token IDs into subsequences of length max_length

    with a specified token overlap between consecutive chunks.
    """
    # [TODO - Add your code here]
    if len(encoded_sequence) <= max_length:
        return [encoded_sequence]

    step = max_length - overlap
    subsequences = []

    for i in range(0, len(encoded_sequence), step):
        chunk = encoded_sequence[i:i + max_length]
        subsequences.append(chunk)
        if i + max_length >= len(encoded_sequence):
            break

    return subsequences

    
Verification: Run the final test blocks to produce the arrays and verify all green checkmarks on the lab assessment page.

Step 4:


def segment_encoded_sequence(
        sequence: list[int],
        max_length: int,
        n_overlap: int
) -> list[list[int]]:
    """Segment a long encoded sequence into overlapping subsequences of maximum
    length.

    Divides the input sequence into chunks of max_length tokens with specified
    overlap between consecutive segments. The final segment may be shorter than
    max_length if insufficient tokens remain.

    Args:
        sequence: List of token indices to segment.
        max_length: Maximum length for each subsequence.
        n_overlap: Number of tokens to overlap between consecutive segments.

    Returns:
        List of subsequences, each with at most max_length token indices. All
        segments except possibly the last will have exactly max_length tokens.
    """
    subsequences = []

    if len(sequence) <= max_length:
        return [sequence]

    step = max_length - n_overlap

    for i in range(0, len(sequence), step):
        chunk = sequence[i:i + max_length]
        subsequences.append(chunk)
        if i + max_length >= len(sequence):
            break

    return subsequences


Step 5:

def create_training_sequences(
        dataset: list[str],
        context_length: int,
        n_overlap: int,
        tokenizer: EnhancedTokenizer
) -> tuple[np.ndarray, np.ndarray]:
    """Create training input-target sequence pairs from text dataset.

    Encodes text data into token sequences, segments them into fixed-length
    overlapping windows, and creates input-target pairs for language modeling
    where targets are inputs shifted by one position.

    Args:
        dataset: List of text strings to process into training sequences.
        context_length: Maximum sequence length for model input.
        n_overlap: Number of tokens to overlap between consecutive segments.
        tokenizer: Tokenizer object with encode method for text-to-tokens
            conversion.

    Returns:
        Tuple of (inputs, targets) where:
        - inputs: Array of token sequences of length context_length.
        - targets: Array of target sequences (inputs shifted by one position).
    """

    segmentation_length = context_length + 1
    # The segments are one token longer than the model's maximum input length,
    # because the target (next) tokens to predict are the input tokens shifted
    # by one position.

    pad_token_id = tokenizer.pad_token_id
    encoded_tokens = []

    # Add your code here to:
    #
    # 1. Iterate over the entries in the dataset.
    # 2. For each dataset entry (text), encode the text into a sequence of token ids.
    # 3. Segment the sequence of token ids into overlapping segments or parts.
    # 4. Include the segments in the list of encoded tokens.
    # 5. Ensure that `encoded_tokens` is a list of lists, where each inner list
    #    represents a sequence of scalars (e.g., integers for tokenized text).
    for text in dataset:
        encoded_seq = tokenizer.encode(text)
        segments = segment_encoded_sequence(encoded_seq, segmentation_length, n_overlap)
        encoded_tokens.extend(segments)

    # Create padded sequences one token longer than the maximum input length.
    padded_sequences = keras.preprocessing.sequence.pad_sequences(
            encoded_tokens,
            maxlen=segmentation_length,
            padding="post",
            value=pad_token_id)

    # Create inputs and targets from padded sequences.
    inputs = padded_sequences[:, :-1]
    targets = padded_sequences[:, 1:]
    return inputs, targets
💡 Troubleshooting & Common Pitfalls
Runtime Disconnects: Colab Enterprise sessions can time out if idle. Check the top-right indicator to verify colab-cpu-runtime is active before executing code cells.

Array Shapes: Ensure targets and inputs match (N, max_length - 1) after accounting for next-token shift offsets.

Pad Tokens: Ensure the default padding ID matches the notebook specification (0 or tokenizer-defined <pad>).
