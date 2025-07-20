🔡 Comparative Study: Deep Learning Models on Masked Morse Code vs Human Language
This project presents a comparative study of deep learning architectures — specifically Bidirectional RNNs, GRUs, and Transformers — to understand how well they model masked sequences in two different symbolic systems:

Morse Code — a dot-dash symbolic language

Human Language — English sentences with masked words

The focus is on evaluating these models’ ability to recover masked tokens and generalize across symbolic vs natural language domains.

📁 Project Files
css
Copy
Edit
📦 Morse-Code-Model-Study
├── Morse_Code_Bidirectional_GRU.ipynb
├── Morse_Code_Bidirectional_RNN.ipynb
├── Morse_Code_Transformer.ipynb
├── Masked_morse_code_dataset.csv
├── Masked NLP Dataset.csv
└── README.md
🎯 Objective
To perform a comparative analysis of deep learning models on:

Learning and recovering masked sequences in symbolic (Morse) vs natural (English) language

Measuring how each model handles:

Sequence continuity and masking

Symbolic sparsity vs natural redundancy

Generalization, attention, and memory capacity

🧠 Models Implemented
Model	Type	Application
Bidirectional RNN	Recurrent	Simple Morse / NLP sequence modeling
Bidirectional GRU	Recurrent (GRU)	Efficient pattern capture
Transformer	Attention-based	Long-range context & masking

📊 Datasets
🟡 Masked_morse_code_dataset.csv
Dot-dash Morse sequences with [MASK] tokens

Used to evaluate symbolic token recovery

🟣 Masked NLP Dataset.csv
English sentences with one or more masked words

Used to mirror BERT-style masking and measure transferability

🔬 Key Research Questions
How do different architectures perform on symbolic vs natural language?

Can attention-based models better handle sparse, rule-based sequences like Morse?

How much context is needed to recover a masked token in each domain?

Do models generalize differently depending on domain structure?

🧬 Model Design: Structuring the Morse Code Transformer
The Transformer model was built from scratch and tailored to the properties of Morse code:

✅ Custom Tokenization
Morse sequences are split by spaces into individual symbols (e.g., ., -, /)

A vocabulary is built including [MASK] and [PAD]

✅ MLM-Style Dataset
True target token is masked

Additional random masks are applied (~15% total masking)

Output labels are set to -100 for non-masked positions to ignore in loss

✅ Architecture Parameters
Embedding Dimension: 256

Attention Heads: 8

Hidden Size: 512

Transformer Layers: 4

Dropout: 0.1

Max Sequence Length: 128

Optimized with CrossEntropyLoss and EarlyStoppingCallback

✅ Training Setup
Batch Size: 32

Epochs: 50

Learning Rate: 3e-4

Device: GPU (if available)

🏁 Getting Started
Installation
bash
Copy
Edit
pip install torch transformers pandas numpy scikit-learn
Running the Code
Open each notebook (RNN / GRU / Transformer)

Choose the appropriate dataset (Masked_morse_code_dataset.csv or Masked NLP Dataset.csv)

Run all cells to train and evaluate

🚀 Future Work
Extend to multi-mask recovery or span prediction

Incorporate causal decoding for generation tasks

Study cross-domain transfer: can training on NLP help Morse?

Visualize attention weights for symbolic alignment

✍️ Author
Urja Damodhar
Machine Learning Research Assistant, Boston University
Portfolio • LinkedIn

