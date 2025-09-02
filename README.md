## BERT Baseline Model

Multi-task classification model for paired inputs (history, response) using Hugging Face Transformers.

### Quick Start

#### Colab
```python
!git clone https://github.com/JayeshAgarwal03/BERT_baselineModel.git
%cd BERT_baselineModel
!pip install -r requirements.txt
!python src/main.py
```

#### Local
```bash
pip install -r requirements.txt
python -m src.main
```

### Structure
- `src/main.py` - Training script
- `src/models/` - Model definition
- `src/utils/` - Data processing
- `src/metrics/` - Evaluation
- `src/config/` - Hyperparameters

### Dataset
Place `dev_testset.json` at `/content/dev_testset.json` or update path in `src/main.py`.

### Config
Edit `src/config/config.py` to modify:
- Model: `MODEL_NAME`, `EPOCHS`, `LEARNING_RATE`
- Data: `BATCH_SIZE`, `MAX_LENGTH`


