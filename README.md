
# Lab 6 – Streaming Language Modeling with Poem Sentiment and GPT-Neo

This lab implements a **streaming language modeling pipeline** using the
[`poem_sentiment`](https://huggingface.co/datasets/poem_sentiment) dataset and
the **GPT-Neo 125M** causal language model.

Instead of loading the whole dataset into memory, the pipeline processes data
as a stream, builds fixed-length token blocks with overlap, and feeds them into
GPT-Neo to compute loss, approximate perplexity, and simple throughput
statistics.

---

## Notebook

- **`LAB6.ipynb`** – Jupyter/Colab notebook containing all the code and
  experiments for this lab.

---

## Pipeline overview

1. **Streaming dataset loading**

   - Uses the Hugging Face `datasets` library.
   - Loads the `poem_sentiment` dataset in **streaming mode**:
     - `verse_text` – short poem text.
     - `label` – sentiment category (0–3).

2. **Custom text format for language modeling**

   - Each example is converted to the form:

     ```text
     <sentiment_LABEL> verse_text
     ```

   - Example:

     ```text
     <sentiment_1> with pale blue berries, in these peaceful shades...
     ```

   - Sentiment is injected as a special token (`<sentiment_0>`, …, `<sentiment_3>`),
     so the label is part of the language-model input.

3. **Tokenization**

   - Uses `AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")`.
   - Sets `pad_token` to the EOS token.
   - Adds four additional special tokens for the sentiment tags.
   - Applies tokenization lazily over the streaming dataset.

4. **Rolling buffer and block creation**

   - Maintains a rolling buffer of token IDs.
   - Builds fixed-length blocks of size `block_size = 128`.
   - Supports **overlap** between blocks via a `stride` parameter
     (e.g., `stride = 96`, so 32 tokens overlap between consecutive blocks).
   - Each block is returned with:
     - `input_ids`
     - `attention_mask`

5. **IterableDataset and DataLoader**

   - Wraps the generator in a custom `StreamingLMIterableDataset`.
   - Uses a PyTorch `DataLoader` with:
     - custom `collate_fn`
     - `batch_size` (e.g., 4)
   - For causal language modeling, `labels` are set equal to `input_ids`.

6. **GPT-Neo model inference**

   - Loads `GPTNeoForCausalLM` from `EleutherAI/gpt-neo-125M`.
   - Resizes token embeddings to include the new sentiment tokens.
   - Moves the model to GPU if available.
   - Runs several streamed batches through the model and records:
     - batch loss values
     - average loss across sampled batches
     - approximate perplexity
     - number of tokens processed and tokens/second

7. **Sample decoded block**

   - Decodes one example block back to text to inspect:
     - presence of `<sentiment_*` tokens
     - overall quality and structure of the tokenized sequence.

---

## How to run (Google Colab)

1. Open **Google Colab** and upload `LAB6.ipynb`, or open it directly from GitHub.
2. (Optional) Enable GPU:
   - `Runtime → Change runtime type → Hardware accelerator: GPU`.
3. Run the setup cell to install dependencies:

   ```python
   !pip install -q "transformers==4.44.2" "datasets==2.21.0" "accelerate==0.34.2"
