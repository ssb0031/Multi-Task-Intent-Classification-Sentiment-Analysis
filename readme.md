# Multi-Task Intent Classification & Sentiment Analysis

A sophisticated multi-phase training pipeline for joint intent classification and sentiment analysis using XLM-RoBERTa with progressive unfreezing, BiLSTM layers, and sentiment-intent fusion mechanisms.

## 📋 Features

- **Multi-Phase Training**: 5-phase progressive training strategy
- **Multi-Task Learning**: Simultaneous intent classification and sentiment analysis
- **Advanced Architecture**: XLM-RoBERTa + BiLSTM + fusion layers
- **Memory Optimization**: Gradient accumulation, AMP, gradient checkpointing
- **Comprehensive Metrics**: Accuracy, F1, fuzzy intent scoring, calibration error (ECE)
- **Stratified Splitting**: Ensures class representation in train/val sets

## 🏗️ Architecture

The model employs a phased approach:
1. **Phase 1**: Frozen backbone, train only heads
2. **Phase 2**: Progressive backbone unfreezing
3. **Phase 3**: Add BiLSTM layer
4. **Phase 4**: Enable sentiment-intent fusion
5. **Phase 5**: Final fine-tuning


## ⚙️ Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (recommended)

Install dependencies:

pip install torch transformers sentence-transformers scikit-learn pandas numpy matplotlib seaborn
🚀 Usage
Prepare your data in augmented_intents.json format

Configure hyperparameters in the script

Run the main script:


python "Intent Classification + Sentiment.py"
📊 Results
The pipeline generates:

Confusion matrices for each phase

Classification reports (intent & sentiment)

Training history metrics

Calibration error analysis

Fuzzy intent scoring

📈 Key Metrics
Intent Accuracy: Multi-class classification accuracy

Sentiment Accuracy: 3-class sentiment classification

Fuzzy Intent Score: Semantic similarity-based scoring

ECE: Expected Calibration Error for confidence calibration

F1 Scores: Macro and weighted averages

🔧 Configuration
Tuned for 16GB VRAM with:

Batch size: 4 (effective 16 with gradient accumulation)

Max sequence length: 64

Learning rate: 3e-5

Weight decay: 0.01

📝 License
MIT License - see LICENSE file for details.


🤝 Contributing
Contributions welcome! Please open an issue or submit a pull request.

