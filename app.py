import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import json
import os

st.set_page_config(page_title="Multilingual NLU System", page_icon="🌍", layout="wide")
st.title("🌍 Multilingual NLU System")
st.markdown("**Intent Classification + Slot Filling for 51 Languages**")

@st.cache_resource
def load_models():
    # Get the current directory (where app.py is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize variables
    intent_tokenizer = None
    intent_model = None
    id2intent = None
    slot_tokenizer = None
    slot_model = None
    id2slot = None
    
    try:
        # Intent model: use checkpoint (config has id2label)
        intent_model_dir = os.path.join(current_dir, 'results_intent_classification', 'checkpoint-71965')
        if not os.path.isdir(intent_model_dir):
            intent_model_dir = os.path.join(current_dir, 'results_intent_classification', 'checkpoint-57572')
        intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_dir)
        intent_model = AutoModelForSequenceClassification.from_pretrained(
            intent_model_dir, torch_dtype=torch.float16
        ).to(device)
        intent_model.eval()
        id2intent = intent_model.config.id2label

        # Slot model: final_model uses id2label.json
        slot_model_dir = os.path.join(current_dir, 'slot_filling_model_crf', 'final_model')
        slot_tokenizer = AutoTokenizer.from_pretrained(slot_model_dir)
        slot_model = AutoModelForTokenClassification.from_pretrained(
            slot_model_dir, torch_dtype=torch.float16
        ).to(device)
        slot_model.eval()
        with open(os.path.join(slot_model_dir, 'id2label.json'), 'r') as f:
            id2slot = json.load(f)
            
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()
    
    return {
        'intent_tokenizer': intent_tokenizer,
        'intent_model': intent_model,
        'id2intent': id2intent,
        'device': device,
        'slot_tokenizer': slot_tokenizer,
        'slot_model': slot_model,
        'id2slot': id2slot
    }

# Show loading indicator
with st.spinner("Loading models... This may take a few minutes for the first time."):
    models = load_models()

@torch.inference_mode()
def predict_intent(utterance):
    try:
        device = models['device']
        tokenizer = models['intent_tokenizer']
        model = models['intent_model']
        id2intent = models['id2intent']
        
        inputs = tokenizer(utterance, return_tensors='pt', truncation=True, 
                          max_length=128, padding='max_length').to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=-1).item()
            confidence = torch.softmax(logits, dim=-1).squeeze()[pred_id].item()
        
        return id2intent[str(pred_id)], confidence
    except Exception as e:
        st.error(f"Error in intent prediction: {str(e)}")
        return "Error", 0.0

@torch.inference_mode()
def extract_slots(utterance):
    try:
        device = models['device']
        tokenizer = models['slot_tokenizer']
        model = models['slot_model']
        id2slot = models['id2slot']
        
        inputs = tokenizer(utterance, return_tensors='pt', truncation=True,
                          max_length=128, padding='max_length').to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
        
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().cpu().numpy())
        
        slots_raw = []
        current_slot_type = None
        current_slot_tokens = []
        
        for token, pred_id in zip(tokens, predictions):
            slot_label = id2slot[str(int(pred_id))]
            
            if token in ['<s>', '</s>', '<pad>']:
                if current_slot_type and current_slot_tokens:
                    slots_raw.append({'type': current_slot_type, 'tokens': current_slot_tokens})
                current_slot_type = None
                current_slot_tokens = []
                continue
            
            if slot_label.startswith('B-'):
                if current_slot_type and current_slot_tokens:
                    slots_raw.append({'type': current_slot_type, 'tokens': current_slot_tokens})
                current_slot_type = slot_label[2:]
                current_slot_tokens = [token]
            elif slot_label.startswith('I-'):
                slot_type = slot_label[2:]
                if slot_type == current_slot_type and current_slot_type:
                    current_slot_tokens.append(token)
                else:
                    if current_slot_type and current_slot_tokens:
                        slots_raw.append({'type': current_slot_type, 'tokens': current_slot_tokens})
                    current_slot_type = slot_type
                    current_slot_tokens = [token]
            elif slot_label == 'O':
                if current_slot_type and current_slot_tokens:
                    slots_raw.append({'type': current_slot_type, 'tokens': current_slot_tokens})
                current_slot_type = None
                current_slot_tokens = []
        
        if current_slot_type and current_slot_tokens:
            slots_raw.append({'type': current_slot_type, 'tokens': current_slot_tokens})
        
        slots_merged = []
        i = 0
        while i < len(slots_raw):
            current = slots_raw[i]
            merged_tokens = list(current['tokens'])
            j = i + 1
            while j < len(slots_raw) and slots_raw[j]['type'] == current['type']:
                if not slots_raw[j]['tokens'][0].startswith('▁'):
                    merged_tokens.extend(slots_raw[j]['tokens'])
                    j += 1
                else:
                    break
            
            slot_value = ''
            for token in merged_tokens:
                if token.startswith('▁'):
                    if slot_value:
                        slot_value += ' '
                    slot_value += token[1:]
                else:
                    slot_value += token
            
            if slot_value.strip():
                slots_merged.append((current['type'], slot_value.strip()))
            i = j
        
        return slots_merged
    except Exception as e:
        st.error(f"Error in slot extraction: {str(e)}")
        return []

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Enter Your Utterance")
    
    examples = [
        "Wake me up at 6 AM tomorrow",
        "Book a table at an Italian restaurant",
        "Play jazz music by Miles Davis",
        "What's the weather in Paris?",
    ]
    
    st.write("**Quick Examples:**")
    cols = st.columns(2)
    for i, example in enumerate(examples):
        if cols[i % 2].button(f"📌 {example[:30]}..."):
            st.session_state.utterance = example
    
    utterance = st.text_area(
        "Type or select an example:",
        value=st.session_state.get('utterance', ''),
        height=100,
        key='utterance'
    )

with col2:
    st.header("System Info")
    st.metric("Supported Languages", "51")
    st.metric("Intent Classes", "60+")
    st.metric("Slot Types", "40+")

if utterance:
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Intent Classification")
        intent, confidence = predict_intent(utterance)
        
        st.metric("Detected Intent", intent)
        st.metric("Confidence", f"{confidence:.2%}")
        
        st.progress(confidence)
    
    with col2:
        st.subheader("📍 Slot Extraction")
        slots = extract_slots(utterance)
        
        if slots:
            for slot_type, slot_value in slots:
                st.write(f"**{slot_type}**: `{slot_value}`")
        else:
            st.write("No slots detected")
    
    st.divider()
    st.subheader("📊 Full Prediction Result")
    
    result = {
        'utterance': utterance,
        'intent': intent,
        'confidence': float(confidence),
        'slots': [(st, sv) for st, sv in slots]
    }
    
    st.json(result)

st.divider()
st.markdown("""
---
**Multilingual NLU System** | Course Project  
Built with XLM-RoBERTa, trained on MASSIVE dataset (20 languages)  
Works on 51 languages with zero-shot cross-lingual transfer
""")