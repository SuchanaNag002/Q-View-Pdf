# =============================================================================
# DistilBERT Fine-tuning Pipeline with Enhanced Epoch Tracking
# Enhanced for PDF Processing and NLP-based Rewriting Only
# =============================================================================

!pip install torch transformers datasets scikit-learn pdfplumber spacy nltk pandas numpy reportlab ipywidgets
!python -m spacy download en_core_web_sm
!python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

import torch
import json
import pandas as pd
import numpy as np
import os
import shutil
import re
import warnings
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from google.colab import files

from transformers import (
    DistilBertTokenizer, DistilBertTokenizerFast, DistilBertForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback, EarlyStoppingCallback, DistilBertConfig
)

from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import pdfplumber
import nltk
import spacy

try:
    nlp_spacy = spacy.load('en_core_web_sm')
except OSError:
    print("Spacy 'en_core_web_sm' model not found. Please run: python -m spacy download en_core_web_sm")
    nlp_spacy = None

warnings.filterwarnings('ignore')

# =============================================================================
# GLOBAL VARIABLES FOR COLAB INTERFACE
# =============================================================================
uploaded_classification_data = None
uploaded_pdf_path = None
trained_classification_model = None

# =============================================================================
# ENHANCED TRAINER CALLBACK FOR EPOCH TRACKING
# =============================================================================
class EpochMetricsCallback(TrainerCallback):
    def __init__(self):
        self.epoch_metrics = []
        self.best_epoch = None
        self.best_metric_value = -float('inf')
        self.metric_name = "eval_accuracy"

    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        if logs and state.epoch is not None:
            current_epoch = int(state.epoch)

            # Extract all available metrics
            epoch_data = {
                'epoch': current_epoch,
                'eval_loss': logs.get('eval_loss', 0.0),
                'eval_accuracy': logs.get('eval_accuracy', 0.0),
                'eval_f1': logs.get('eval_f1', 0.0),
                'eval_precision': logs.get('eval_precision', 0.0),
                'eval_recall': logs.get('eval_recall', 0.0),
                'train_loss': getattr(state, 'log_history', [{}])[-1].get('train_loss', 0.0) if hasattr(state, 'log_history') and state.log_history else 0.0
            }

            # Check if this epoch already exists (to avoid duplicates)
            existing_epoch = next((item for item in self.epoch_metrics if item['epoch'] == current_epoch), None)
            if existing_epoch:
                existing_epoch.update(epoch_data)
            else:
                self.epoch_metrics.append(epoch_data)

            # Display current epoch metrics
            print(f"\n{'='*60}")
            print(f"📊 EPOCH {current_epoch} RESULTS")
            print(f"{'='*60}")
            print(f"🔹 Training Loss:    {epoch_data['train_loss']:.4f}")
            print(f"🔹 Validation Loss:  {epoch_data['eval_loss']:.4f}")
            print(f"🔹 Accuracy:         {epoch_data['eval_accuracy']:.4f} ({epoch_data['eval_accuracy']*100:.2f}%)")
            print(f"🔹 F1 Score:         {epoch_data['eval_f1']:.4f}")
            print(f"🔹 Precision:        {epoch_data['eval_precision']:.4f}")
            print(f"🔹 Recall:           {epoch_data['eval_recall']:.4f}")

            # Track best epoch
            current_metric_value = epoch_data[self.metric_name]
            if current_metric_value > self.best_metric_value:
                self.best_metric_value = current_metric_value
                self.best_epoch = current_epoch
                print(f"✨ NEW BEST EPOCH! (Accuracy: {current_metric_value:.4f})")
            else:
                print(f"💡 Best so far: Epoch {self.best_epoch} (Accuracy: {self.best_metric_value:.4f})")

            print(f"{'='*60}")

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n{'🎯'*20}")
        print("TRAINING COMPLETED - FINAL SUMMARY")
        print(f"{'🎯'*20}")

        if self.epoch_metrics:
            # Display all epoch results in a table format
            print("\n📋 ALL EPOCHS SUMMARY:")
            print("-" * 90)
            print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<10} {'Accuracy':<10} {'F1':<8} {'Precision':<11} {'Recall':<8}")
            print("-" * 90)

            for metrics in self.epoch_metrics:
                marker = "⭐" if metrics['epoch'] == self.best_epoch else "  "
                print(f"{marker} {metrics['epoch']:<6} {metrics['train_loss']:<12.4f} {metrics['eval_loss']:<10.4f} "
                      f"{metrics['eval_accuracy']:<10.4f} {metrics['eval_f1']:<8.4f} "
                      f"{metrics['eval_precision']:<11.4f} {metrics['eval_recall']:<8.4f}")

            print("-" * 90)
            print(f"\n🏆 BEST PERFORMING EPOCH: {self.best_epoch}")
            best_metrics = next(item for item in self.epoch_metrics if item['epoch'] == self.best_epoch)
            print(f"   📈 Accuracy: {best_metrics['eval_accuracy']:.4f} ({best_metrics['eval_accuracy']*100:.2f}%)")
            print(f"   📈 F1 Score: {best_metrics['eval_f1']:.4f}")
            print(f"   📈 Precision: {best_metrics['eval_precision']:.4f}")
            print(f"   📈 Recall: {best_metrics['eval_recall']:.4f}")
            print(f"   📉 Validation Loss: {best_metrics['eval_loss']:.4f}")

        print(f"\n{'🎯'*20}")

# =============================================================================
# PDF PROCESSOR
# =============================================================================
class PdfProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        if not nlp_spacy:
            raise RuntimeError("SpaCy model 'en_core_web_sm' not loaded. Cannot proceed with PDF processing.")

    def extract_text_chunks(self, min_chunk_length=20, max_chunk_length=512):
        """Extract text from PDF and segment it into meaningful chunks"""
        full_text = ""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
        except Exception as e:
            print(f"Error reading PDF {self.pdf_path}: {e}")
            return []

        # Basic cleaning
        full_text = re.sub(r'\s*\n\s*', '\n', full_text)
        full_text = re.sub(r'(\w)-\n(\w)', r'\1\2', full_text)
        full_text = re.sub(r' +', ' ', full_text)

        doc = nlp_spacy(full_text)
        sentences = [sent.text.strip() for sent in doc.sents]

        potential_questions = []
        current_block = ""

        question_starters_regex = r"^(Example \d+|Ex \d+\.\d+|[Qq]uestion \d+[:\.]?|\d+[:\.]\s+|What is|What will be|Find|Calculate|How many|If|Explain|Why|Convert|Tell|The population|Arun bought|I buy|Juhi sells|Amina buys)"
        solution_indicator_regex = r"^(Solution|Answer|Sol[:\.]?$|Ans[:\.]?$)"

        for text in sentences:
            if re.search(solution_indicator_regex, text.strip(), re.IGNORECASE):
                if current_block and len(current_block) > min_chunk_length:
                    potential_questions.append(current_block.strip())
                current_block = ""
                continue

            if re.search(question_starters_regex, text.strip(), re.IGNORECASE) or \
               (current_block and not text.strip().endswith('.') and not text.strip().endswith('?')):
                if current_block:
                    current_block += " " + text
                else:
                    current_block = text
            else:
                if current_block:
                     if len(current_block) > min_chunk_length:
                        potential_questions.append(current_block.strip())
                     current_block = ""
                if (text.endswith('?') or re.match(r"^(What|How|When|Why|Who|Which|Find|Calculate|Derive|Explain)\b", text, re.IGNORECASE)) \
                   and len(text) > min_chunk_length:
                    potential_questions.append(text)

        if current_block and len(current_block) > min_chunk_length:
            potential_questions.append(current_block.strip())

        # Filter and clean chunks
        processed_chunks = []
        for chunk in potential_questions:
            chunk = re.sub(r"Page \d+", "", chunk, flags=re.IGNORECASE)
            chunk = re.sub(r"Chapter \d+", "", chunk, flags=re.IGNORECASE)
            chunk = re.sub(r"^\s*\d+\s*$", "", chunk)
            chunk = chunk.strip()
            if min_chunk_length <= len(chunk) <= max_chunk_length:
                processed_chunks.append(chunk)

        # Remove duplicates while preserving order
        seen = set()
        unique_chunks = [x for x in processed_chunks if not (x in seen or seen.add(x))]

        print(f"Extracted {len(unique_chunks)} potential question chunks from PDF.")
        return unique_chunks

# =============================================================================
# ENHANCED DISTILBERT TRAINER (Classification Only)
# =============================================================================
class DistilBERTClassificationTrainer:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

        # FIXED: Initialize model with dropout configuration
        config = DistilBertConfig.from_pretrained(model_name)
        config.dropout = 0.3
        config.attention_dropout = 0.1
        config.num_labels = 2

        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        self.epoch_callback = EpochMetricsCallback()

        print(f"Tokenizer vocab size: {len(self.tokenizer)}")
        print(f"Model type: {type(self.model).__name__}")
        print(f"Task type: classification")
        print(f"Dropout rate: {config.dropout}")

    def prepare_classification_data(self, data_source):
        """Prepare data for sequence classification"""
        if isinstance(data_source, str):
            with open(data_source, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        elif isinstance(data_source, list):
            raw_data = data_source
        else:
            raise ValueError("data_source must be a list of dicts or a file path to a JSON file.")

        processed = []
        for item in raw_data:
            if 'text' in item and 'label' in item:
                processed.append({
                    'text': str(item['text']),
                    'label': int(item['label'])
                })
        return processed

    def tokenize_classification_function(self, examples):
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        tokenized['labels'] = examples['label']
        return tokenized

    def compute_classification_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

    def train_model(self, train_data_source, eval_data_source=None, training_args_dict=None):
      """Train the DistilBERT classification model with enhanced epoch tracking"""
      _train_data = self.prepare_classification_data(train_data_source)
      _eval_data = self.prepare_classification_data(eval_data_source) if eval_data_source else None

      if not _train_data:
          print("No training data loaded. Aborting training.")
          return None

      train_df = pd.DataFrame(_train_data)
      train_dataset = Dataset.from_pandas(train_df)

      if _eval_data:
          eval_df = pd.DataFrame(_eval_data)
          eval_dataset = Dataset.from_pandas(eval_df)
      else:
          if len(train_dataset) < 2:
              eval_dataset = train_dataset
          else:
              # FIXED: Remove stratify_by_column parameter to fix the error
              split_dataset = train_dataset.train_test_split(test_size=0.2, seed=42)
              train_dataset = split_dataset['train']
              eval_dataset = split_dataset['test']

      tokenized_train_dataset = train_dataset.map(self.tokenize_classification_function, batched=True, remove_columns=['text'])
      tokenized_eval_dataset = eval_dataset.map(self.tokenize_classification_function, batched=True, remove_columns=['text'])

      # FIXED: Updated training arguments to prevent overfitting
      default_training_args = {
          "output_dir": "./distilbert-classification-checkpoints",
          "eval_strategy": "epoch",
          "save_strategy": "epoch",
          "logging_strategy": "epoch",
          "num_train_epochs": 3,  # FIXED: Reduced to 3 epochs as requested
          "per_device_train_batch_size": 16,  # FIXED: Increased batch size
          "per_device_eval_batch_size": 16,
          "gradient_accumulation_steps": 1,
          "warmup_steps": 50,  # FIXED: Reduced warmup steps
          "eval_steps": 50,
          "logging_steps": 50,
          "save_steps": 50,
          "learning_rate": 2e-5,  # FIXED: Reduced learning rate
          "weight_decay": 0.1,  # FIXED: Increased weight decay for regularization
          "logging_dir": "./logs",
          "save_total_limit": 3,  # FIXED: Only keep 3 checkpoints
          "load_best_model_at_end": True,
          "metric_for_best_model": "eval_accuracy",
          "greater_is_better": True,
          "report_to": "none",
          "dataloader_pin_memory": False,
          "remove_unused_columns": True,
          "fp16": torch.cuda.is_available(),
          "push_to_hub": False,
          # FIXED: Add dropout and other regularization
          "dataloader_drop_last": True,  # Drop incomplete batches
          "seed": 42,  # Set seed for reproducibility
      }

      if training_args_dict:
          default_training_args.update(training_args_dict)

      training_args = TrainingArguments(**default_training_args)
      data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")

      # FIXED: Adjusted early stopping to be less aggressive
      early_stopping_callback = EarlyStoppingCallback(
          early_stopping_patience=3,  # FIXED: Increased patience
          early_stopping_threshold=0.01  # FIXED: Increased threshold
      )

      trainer = Trainer(
          model=self.model,
          args=training_args,
          train_dataset=tokenized_train_dataset,
          eval_dataset=tokenized_eval_dataset,
          data_collator=data_collator,
          tokenizer=self.tokenizer,
          compute_metrics=self.compute_classification_metrics,
          callbacks=[self.epoch_callback, early_stopping_callback]
      )

      print("🚀 Starting classification training with epoch-by-epoch tracking...")
      print(f"📊 Training will run for up to {training_args.num_train_epochs} epochs")
      print(f"🎯 Early stopping patience: 3 epochs")
      print(f"📚 Training samples: {len(train_dataset)}")
      print(f"🔍 Validation samples: {len(eval_dataset)}")
      print("-" * 60)

      trainer.train()

      # Save the best model
      final_model_path = "./distilbert-classification-final"
      if os.path.exists(final_model_path):
          shutil.rmtree(final_model_path)

      trainer.save_model(final_model_path)
      self.tokenizer.save_pretrained(final_model_path)

      print(f"\n💾 BEST MODEL SAVED!")
      print(f"📁 Location: {final_model_path}")
      print(f"🏆 Based on: Epoch {self.epoch_callback.best_epoch}")
      print(f"📈 Best Accuracy: {self.epoch_callback.best_metric_value:.4f}")

      # Save epoch metrics for reference
      metrics_file = os.path.join(final_model_path, "epoch_metrics.json")
      with open(metrics_file, 'w') as f:
          json.dump({
              'all_epochs': self.epoch_callback.epoch_metrics,
              'best_epoch': self.epoch_callback.best_epoch,
              'best_accuracy': self.epoch_callback.best_metric_value
          }, f, indent=2)

      print(f"📊 Epoch metrics saved to: {metrics_file}")

      return trainer

# =============================================================================
# ENHANCED NLP REWRITER CLASS
# =============================================================================
class DistilBERTNLPRewriter:
    def __init__(self, classification_model_path):
        if not os.path.exists(classification_model_path):
            raise FileNotFoundError(f"Classification model not found at {classification_model_path}")

        self.classifier_tokenizer = DistilBertTokenizerFast.from_pretrained(classification_model_path)
        self.classifier = DistilBertForSequenceClassification.from_pretrained(classification_model_path)
        self.classifier.eval()

        # Load and display model info
        metrics_file = os.path.join(classification_model_path, "epoch_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
                print(f"🤖 Loaded model from best epoch: {metrics_data['best_epoch']}")
                print(f"📈 Model accuracy: {metrics_data['best_accuracy']:.4f}")

    def classify_text(self, text):
        inputs = self.classifier_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        return "question" if prediction == 1 else "statement"

    def simple_rewrite_formal(self, text):
        """Convert question format to statement format"""
        rewrite_rules = {
            r'\bWhat is\b': 'Calculate', r'\bwhat is\b': 'calculate',
            r'\bWhat will be\b': 'Determine', r'\bwhat will be\b': 'determine',
            r'\bWhat are\b': 'Find', r'\bwhat are\b': 'find',
            r'\bFind\b': 'Calculate', r'\bfind\b': 'calculate',
            r'\bWork out\b': 'Calculate', r'\bwork out\b': 'calculate',
            r'\bHow much\b': 'Calculate', r'\bhow much\b': 'calculate',
            r'\bHow many\b': 'Count', r'\bhow many\b': 'count',
            r'\?': '.',
        }
        rewritten = text
        for pattern, replacement in rewrite_rules.items():
            rewritten = re.sub(pattern, replacement, rewritten)
        return rewritten.strip()

    def advanced_nlp_rewrite(self, text, change_factor=1.2, num_increment=2):
        """Enhanced NLP-based rewriting with better math object substitution"""
        if not nlp_spacy:
            return text

        doc = nlp_spacy(text)
        rewritten_tokens = [token.text for token in doc]
        changed = False

        # Enhanced math object substitutions
        math_object_nouns = {
            # Geometric shapes
            "triangle": "rectangle", "rectangle": "triangle", "square": "circle",
            "circle": "square", "cube": "cylinder", "cylinder": "cube",
            "sphere": "cone", "cone": "sphere", "pentagon": "hexagon",

            # Measurements
            "radius": "diameter", "diameter": "radius", "height": "width",
            "width": "length", "length": "height", "area": "perimeter",
            "perimeter": "area", "volume": "surface area",

            # Physical quantities
            "speed": "velocity", "velocity": "acceleration", "distance": "displacement",
            "time": "duration", "mass": "weight", "weight": "mass",

            # Mathematical operations
            "sum": "product", "product": "sum", "difference": "quotient",
            "quotient": "difference", "average": "median", "median": "mode",

            # Objects and items
            "apples": "oranges", "oranges": "mangoes", "mangoes": "bananas",
            "bananas": "grapes", "books": "notebooks", "pens": "pencils",
            "chairs": "tables", "cars": "bikes", "houses": "buildings",

            # Currency and units
            "rupees": "dollars", "dollars": "euros", "cents": "pence",
            "cm": "inches", "inches": "feet", "meters": "yards",
            "kg": "pounds", "grams": "ounces", "litres": "gallons",

            # People and roles
            "students": "teachers", "workers": "employees", "boys": "girls",
            "men": "women", "children": "adults",

            # Time units
            "years": "months", "months": "weeks", "weeks": "days",
            "hours": "minutes", "minutes": "seconds",

            # Business terms
            "price": "cost", "cost": "expense", "salary": "income",
            "profit": "loss", "revenue": "expenditure", "discount": "markup",

            # Other common terms
            "matches": "games", "items": "objects", "pieces": "units",
            "lakhs": "thousands", "crores": "millions", "sweets": "candies"
        }

        # Process tokens
        for i, token in enumerate(doc):
            # Handle numbers
            if token.like_num:
                try:
                    original_value = float(token.text)
                    if original_value == 0:
                        new_value = float(num_increment)
                    elif abs(original_value) < 10 and "." not in token.text:
                        # For small integers, add increment
                        new_value = int(original_value + num_increment) if original_value > 0 else int(original_value - num_increment)
                    else:
                        # For larger numbers, apply factor
                        new_value = original_value * change_factor

                    # Format output appropriately
                    if "." not in token.text and "e" not in token.text.lower():
                        rewritten_tokens[i] = str(int(round(new_value)))
                    else:
                        rewritten_tokens[i] = f"{new_value:.2f}"
                    changed = True
                except ValueError:
                    pass

            # Handle nouns and proper nouns
            elif token.pos_ in ("NOUN", "PROPN"):
                lower_token_text = token.text.lower()
                if lower_token_text in math_object_nouns:
                    replacement = math_object_nouns[lower_token_text]
                    # Preserve capitalization
                    if token.text[0].isupper():
                        rewritten_tokens[i] = replacement[0].upper() + replacement[1:]
                    else:
                        rewritten_tokens[i] = replacement
                    changed = True

        if changed:
            # Reconstruct text with proper spacing
            return "".join([tok + (sp if sp else "") for tok, sp in zip(rewritten_tokens, [t.whitespace_ for t in doc])]).strip()
        else:
            return text

    def generate_multiple_variants(self, text, num_variants=3):
        """Generate multiple rewritten variants of the same question"""
        variants = []

        # Variant 1: Formal rewrite
        formal_variant = self.simple_rewrite_formal(text)
        variants.append({"type": "formal", "text": formal_variant})

        # Variant 2: NLP rewrite with default parameters
        nlp_variant_1 = self.advanced_nlp_rewrite(text)
        variants.append({"type": "nlp_default", "text": nlp_variant_1})

        # Variant 3: NLP rewrite with different parameters
        nlp_variant_2 = self.advanced_nlp_rewrite(text, change_factor=1.5, num_increment=3)
        variants.append({"type": "nlp_enhanced", "text": nlp_variant_2})

        # Additional variants if requested
        if num_variants > 3:
            for i in range(3, num_variants):
                factor = 1.1 + (i * 0.2)
                increment = i
                additional_variant = self.advanced_nlp_rewrite(text, change_factor=factor, num_increment=increment)
                variants.append({"type": f"nlp_variant_{i+1}", "text": additional_variant})

        return variants

# =============================================================================
# COLAB INTERFACE FUNCTIONS
# =============================================================================

def upload_classification_data():
    global uploaded_classification_data
    print("Upload your classification training data (JSON format)")
    print("Expected format: [{'text': 'question text', 'label': 1}, {'text': 'statement text', 'label': 0}]")
    uploaded = files.upload()

    if uploaded:
        filename = list(uploaded.keys())[0]
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                uploaded_classification_data = json.load(f)
            print(f"✅ Successfully loaded {len(uploaded_classification_data)} classification samples")

            # Show sample data
            if len(uploaded_classification_data) > 0:
                print("\n📋 Sample data:")
                for i, sample in enumerate(uploaded_classification_data[:3]):
                    print(f"  {i+1}. Text: '{sample.get('text', 'N/A')[:50]}...'")
                    print(f"     Label: {sample.get('label', 'N/A')} ({'Question' if sample.get('label') == 1 else 'Statement'})")

        except Exception as e:
            print(f"❌ Error loading classification data: {e}")

def train_classification_model():
    global uploaded_classification_data, trained_classification_model

    if uploaded_classification_data is None:
        print("❌ Please upload classification data first!")
        return

    print("🚀 Starting enhanced classification model training...")
    print("📊 This will show detailed metrics for each epoch and select the best model!")
    try:
        trainer = DistilBERTClassificationTrainer()
        trained_classification_model = trainer.train_model(uploaded_classification_data)
        print("\n✅ Classification model training completed!")
        print("🏆 Best epoch model has been automatically saved!")
    except Exception as e:
        print(f"❌ Error during classification training: {e}")

def upload_and_process_pdf():
    global uploaded_pdf_path

    if not os.path.exists("./distilbert-classification-final"):
        print("❌ Please train the classification model first!")
        return

    print("Upload your PDF file for processing:")
    uploaded = files.upload()

    if uploaded:
        filename = list(uploaded.keys())[0]
        uploaded_pdf_path = filename

        try:
            print("🔍 Processing PDF...")
            pdf_processor = PdfProcessor(uploaded_pdf_path)
            text_chunks = pdf_processor.extract_text_chunks()

            if not text_chunks:
                print("❌ No text chunks extracted from PDF.")
                return

            print("🤖 Analyzing and rewriting content with best model...")
            rewriter = DistilBERTNLPRewriter(
                classification_model_path="./distilbert-classification-final"
            )

            results = []
            questions_count = 0
            statements_count = 0

            for i, chunk in enumerate(text_chunks):
                if len(chunk) < 10:
                    continue

                classification = rewriter.classify_text(chunk)
                result = {
                    'chunk_id': i+1,
                    'original_text': chunk,
                    'classification': classification
                }

                if classification == "question":
                    questions_count += 1
                    variants = rewriter.generate_multiple_variants(chunk)
                    result['variants'] = variants
                else:
                    statements_count += 1

                results.append(result)

            # Display results
            print(f"\n{'='*80}")
            print("📊 PROCESSING RESULTS")
            print(f"{'='*80}")
            print(f"📈 Summary: {questions_count} questions, {statements_count} statements found")
            print(f"{'='*80}")

            for result in results:
                print(f"\n🔸 Chunk {result['chunk_id']} ({result['classification'].upper()}):")
                print(f"Original: {result['original_text']}")

                if result['classification'] == "question" and 'variants' in result:
                    print("📝 Generated Variants:")
                    for j, variant in enumerate(result['variants'], 1):
                        print(f"  {j}. [{variant['type']}]: {variant['text']}")

                print("-" * 60)

            # Save results to JSON
            with open('processing_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to 'processing_results.json'")
            print(f"📋 Total processed: {len(results)} chunks")

        except Exception as e:
            print(f"❌ Error processing PDF: {e}")

# =============================================================================
# MAIN COLAB INTERFACE
# =============================================================================

def create_colab_interface():
    print("🎯 DistilBERT Classification & NLP Rewriting Pipeline")
    print("="*60)

    # Create buttons
    upload_class_btn = widgets.Button(
        description='📤 Upload Classification Data',
        style={'button_color': 'lightblue'},
        layout=widgets.Layout(width='300px', height='45px')
    )

    train_class_btn = widgets.Button(
        description='🚀 Train Classification Model',
        style={'button_color': 'orange'},
        layout=widgets.Layout(width='300px', height='45px')
    )

    process_pdf_btn = widgets.Button(
        description='📄 Upload & Process PDF',
        style={'button_color': 'lightcoral'},
        layout=widgets.Layout(width='300px', height='45px')
    )

    # Attach functions to buttons
    upload_class_btn.on_click(lambda x: upload_classification_data())
    train_class_btn.on_click(lambda x: train_classification_model())
    process_pdf_btn.on_click(lambda x: upload_and_process_pdf())

    # Create layout
    step1 = widgets.VBox([
        widgets.HTML("<h3>📚 Step 1: Upload Training Data</h3>"),
        widgets.HTML("<p>Upload JSON file with question/statement classification data</p>"),
        upload_class_btn
    ])

    step2 = widgets.VBox([
        widgets.HTML("<h3>🔧 Step 2: Train Classification Model</h3>"),
        widgets.HTML("<p>Train DistilBERT to classify questions vs statements</p>"),
        train_class_btn
    ])

    step3 = widgets.VBox([
        widgets.HTML("<h3>📖 Step 3: Process PDF</h3>"),
        widgets.HTML("<p>Upload PDF and generate multiple rewritten variants using NLP</p>"),
        process_pdf_btn
    ])

    interface = widgets.VBox([
        widgets.HTML("<div style='text-align: center; font-size: 20px; font-weight: bold; margin-bottom: 20px; color: #2E86AB;'>🎯 DistilBERT NLP Pipeline</div>"),
        widgets.HTML("<div style='text-align: center; margin-bottom: 30px; color: #666;'>Classification + Advanced NLP Rewriting</div>"),
        step1,
        widgets.HTML("<br>"),
        step2,
        widgets.HTML("<br>"),
        step3,
        widgets.HTML("<br><div style='text-align: center; color: gray; font-style: italic;'>✨ Follow the steps in order for best results ✨</div>")
    ])

    display(interface)

# =============================================================================
# RUN THE INTERFACE
# =============================================================================

if __name__ == "__main__":
    create_colab_interface()
