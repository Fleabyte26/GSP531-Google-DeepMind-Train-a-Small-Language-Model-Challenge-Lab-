# GSP531: Build Your Own Small Language Model — Challenge Lab Solution

A complete walkthrough, script solutions, and troubleshooting guide for **Google Cloud Challenge Lab GSP531: Develop a Chatbot for the Arabic-Speaking Market**.

---

## 📋 Lab Overview
* **Lab ID:** `GSP531`
* **Track:** *Google DeepMind: 01 Build Your Own Small Language Model*
* **Core Topics Tested:**
  * **Character Tokenizer:** Encode and decode Arabic text at the character level.
  * **N-gram Text Generator:** Build an autoregressive generation loop supporting greedy and random sampling.
  * **Data Pipeline:** Segment sequences with overlapping windows and format input/target arrays for training.

---

## 🛠️ Step-by-Step Implementation

### Task 1: Environment Setup & Notebook Import

1. In the Google Cloud Console, navigate to:
   ```text
   Agent Platform > Notebooks > Colab Enterprise > My notebooks
   ```
2. Set the designated lab **Region** and click **Import**.
3. Select **Cloud Storage**, provide the `notebook_file_path` assigned by your lab environment, and import the notebook.
4. Click the connection menu in the upper right, choose **Connect to an existing runtime**, and select `colab-cpu-runtime`.
5. Execute the initial cells to install packages and load the dataset.

> **Important Note on Kernel Restart:**  
> When running the cell containing `app.kernel.do_shutdown(True)`, Colab will notify you that the kernel crashed or restarted. This is expected. Do **not** re-run that restart cell; simply continue to the next cell (`# Packages used.`) to resume execution.

---

### Task 2: Configure the Character Tokenizer

In the `SimpleArabicCharacterTokenizer` class cell, complete the methods to tokenize and join characters:

```python
class SimpleArabicCharacterTokenizer:
    def __init__(self):
        pass

    def character_tokenize(self, text: str) -> list[str]:
        """Converts raw Arabic text into a list of single-character tokens."""
        # [TODO - Add your code here]
        return list(text)

    def join_text(self, tokens: list[str]) -> str:
        """Joins a list of character tokens back into a single string without padding."""
        # [TODO - Add your code here]
        return "".join(tokens)
```

Run the test cell directly beneath it and verify Task 2 in the lab console.

---

### Task 3: Build N-gram Text Generation

In the `generate_text_from_ngram_model` function cell, implement context window extraction and sampling modes:

```python
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
```

#### Task 3 Test Cell Fix
If the test cell fails with `NameError: name 'tokenizer' is not defined`, ensure `tokenizer` is instantiated before calling `build_ngram_model`:

```python
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
```

---

### Task 4: Prepare Dataset for Training

#### Part 1: Sequence Segmentation (`segment_encoded_sequence`)

Chunk the sequence using `max_length` with step size `max_length - n_overlap`:

```python
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
```

#### Part 2: Training Sequences & Shifting (`create_training_sequences`)

Encode each story, segment it, pad the sequences using Keras, and split into inputs and targets ($y_t = x_{t+1}$):

```python
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

    # Iterate over dataset, encode, and segment into overlapping parts
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
```

---

## 💡 Troubleshooting Checklist
* **Missing Checkmarks:** Make sure to execute the cell containing `logger.info(...)` located immediately below each task. The assessment scripts track these logs to award credit.
* **Kernel Disconnects:** If idle, Colab Enterprise may sleep. Verify the top-right status bar shows `colab-cpu-runtime` is active before running downstream cells.
