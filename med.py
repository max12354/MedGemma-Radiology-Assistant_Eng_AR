import os

# IMPORTANT: Comment out or remove CUDA_LAUNCH_BLOCKING for normal operation as it slows down execution.
# It's useful for detailed CUDA debugging but not for a final version.
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 

os.environ["TORCH_COMPILE_DISABLE"] = "1" 

import torch
import pydicom
import gradio as gr
from PIL import Image
from transformers import pipeline, AutoTokenizer
import traceback
import nltk
import re 

# --- NLTK Data Check and Download ---
def ensure_nltk_punkt():
    nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.append(nltk_data_path)
    
    try:
        nltk.data.find('tokenizers/punkt')
        print("INFO: NLTK 'punkt' tokenizer found in known paths and confirmed working.")
        # Test it briefly
        nltk.sent_tokenize("NLTK test. Sentence splitting active.")
        return True
    except LookupError: 
        print("INFO: NLTK 'punkt' tokenizer not found. Attempting to download...")
        try:
            nltk.download('punkt', quiet=False, download_dir=nltk_data_path) 
            print(f"INFO: 'punkt' downloaded to {nltk_data_path}.")
            # Verify again after download
            nltk.data.find('tokenizers/punkt') 
            nltk.sent_tokenize("NLTK test. Sentence splitting active.")
            print("INFO: NLTK 'punkt' tokenizer confirmed working after download.")
            return True
        except Exception as e_download:
            print(f"WARNING: Error downloading or confirming NLTK 'punkt': {e_download}")
            print("WARNING: NLTK sentence tokenization might fail. Please ensure 'punkt' is correctly installed.")
            print("You can try manually: import nltk; nltk.download('punkt') in a Python interpreter.")
            return False
    except Exception as e_find:
        print(f"WARNING: Generic error finding/confirming NLTK 'punkt': {e_find}")
        return False

NLTK_PUNKT_AVAILABLE = ensure_nltk_punkt()

# --- Device Check ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Device Check ---")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"Detected GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"Selected device for script: {device}")
print(f"--------------------")

try:
    import nibabel as nib
    has_nibabel = True
except ImportError:
    has_nibabel = False
    print("WARNING: nibabel not installed. MRI (.nii, .nii.gz) support will be disabled.")

model_id = "google/medgemma-4b-it"
translation_model_id = "Helsinki-NLP/opus-mt-en-ar"
med_pipe = None
mt_pipe = None
translation_tokenizer = None 

try:
    print(f"INFO: Loading MedGemma model '{model_id}' onto device: {device}...")
    med_pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32},
        device=device,
    )
    print(f"INFO: MedGemma model loaded successfully on {device}.")
except Exception as e:
    print(f"ERROR: Error loading MedGemma model: {e}"); traceback.print_exc()
    print("CRITICAL: MedGemma functionality will be impacted.")

try:
    translation_device_arg = 0 if device == "cuda" else -1
    print(f"INFO: Loading Translation model '{translation_model_id}' onto device argument: {translation_device_arg}")
    mt_pipe = pipeline("translation_en_to_ar", model=translation_model_id, device=translation_device_arg)
    translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_id)
    print(f"INFO: Translation model and tokenizer loaded successfully on {'cuda:0' if translation_device_arg == 0 else 'cpu'}.")

    if mt_pipe and translation_tokenizer: # Test after both are loaded
        try:
            # print("INFO: Testing translation pipe with a simple sentence...") # Can be commented out for final
            # test_sentence = "Hello world."
            # test_translation = mt_pipe(test_sentence, max_new_tokens=60) 
            # print(f"Test translation of '{test_sentence}': {test_translation[0]['translation_text']}")
            pass # Test passed previously
        except Exception as e_test_mt:
            print(f"WARNING: Error during standalone translation test: {e_test_mt}")
except Exception as e:
    print(f"ERROR: Error loading translation model or its tokenizer: {e}"); traceback.print_exc()
    print("WARNING: Translation functionality will be disabled.")

def bilingual(en_text, ar_text):
    return f"{en_text} / {ar_text}"

def load_image_from_path(file_path):
    try:
        if file_path.lower().endswith(".dcm"):
            ds = pydicom.dcmread(file_path); img_array = ds.pixel_array
            if ds.PhotometricInterpretation == "MONOCHROME1": img_array = img_array.max() - img_array
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                wc_val = ds.WindowCenter; ww_val = ds.WindowWidth
                wc = float(wc_val[0] if isinstance(wc_val, pydicom.multival.MultiValue) else wc_val)
                ww = float(ww_val[0] if isinstance(ww_val, pydicom.multival.MultiValue) else ww_val)
                img_min = wc - ww / 2.0; img_max = wc + ww / 2.0
                img_array = img_array.astype(float)
                img_array[img_array < img_min] = img_min; img_array[img_array > img_max] = img_max
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-5) * 255.0
            pil_image = Image.fromarray(img_array.astype('uint8')).convert("RGB")
            return pil_image
        elif has_nibabel and file_path.lower().endswith((".nii", ".nii.gz")):
            img_nib = nib.load(file_path); img_nib_canonical = nib.as_closest_canonical(img_nib)
            img_data = img_nib_canonical.get_fdata()
            if img_data.ndim == 2: slice_data = img_data
            elif img_data.ndim == 3: slice_data = img_data[:, :, img_data.shape[2] // 2]
            elif img_data.ndim == 4: slice_data = img_data[:, :, img_data.shape[2] // 2, img_data.shape[3] // 2]
            else: raise ValueError(f"Unsupported NIfTI image dimension: {img_data.ndim}")
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-5) * 255
            pil_image = Image.fromarray(slice_data.astype('uint8')).convert("RGB")
            return pil_image
        else:
            pil_image = Image.open(file_path).convert("RGB")
            return pil_image
    except Exception as e:
        print(f"ERROR: Error loading image {file_path}: {e}"); traceback.print_exc(); raise

def simple_splitter_by_paragraph_and_length(text, max_chars_per_part=800): # Chars, not tokens
    """A fallback splitter: by paragraph, then by max chars, trying to respect sentence ends."""
    parts = []
    paragraphs = text.split('\n')
    for paragraph_text in paragraphs:
        paragraph_text = paragraph_text.strip()
        if not paragraph_text:
            if parts and parts[-1] != "\n\n": 
                parts.append("\n\n") 
            continue
        
        current_paragraph_part = paragraph_text
        while len(current_paragraph_part) > max_chars_per_part:
            split_point = -1
            # Search backwards from max_chars_per_part for a sentence-ending punctuation
            # Limit search to avoid excessively small initial splits if no punctuation found early
            search_end_idx = max(0, max_chars_per_part - 200) 
            for i in range(min(len(current_paragraph_part)-1, max_chars_per_part), search_end_idx , -1):
                if current_paragraph_part[i] in ".!?":
                    split_point = i + 1
                    break
            if split_point == -1 or split_point == 0: # No suitable punctuation or at the very beginning
                # If no good split point, take a hard cut or look for comma/space
                search_end_idx_space = max(0, max_chars_per_part - 50)
                for i in range(min(len(current_paragraph_part)-1, max_chars_per_part), search_end_idx_space , -1):
                    if current_paragraph_part[i] in " ,": # Prefer space or comma
                        split_point = i + 1
                        break
                if split_point == -1 or split_point == 0:
                    split_point = max_chars_per_part # Force split at max_chars

            parts.append(current_paragraph_part[:split_point].strip())
            current_paragraph_part = current_paragraph_part[split_point:].strip()
        
        if current_paragraph_part: # Add the remaining part of the paragraph
            parts.append(current_paragraph_part)
    
    # Consolidate parts, handling the added paragraph breaks correctly
    final_parts = []
    current_text_block = ""
    for p_idx, p_val in enumerate(parts):
        if p_val == "\n\n":
            if current_text_block: # Add accumulated text before the paragraph break
                final_parts.append(current_text_block)
                current_text_block = ""
            if not final_parts or final_parts[-1] != "\n\n": # Avoid double paragraph breaks
                final_parts.append(p_val) 
        else:
            if current_text_block and not current_text_block.endswith("\n\n"):
                current_text_block += " " + p_val # Add space if joining normal text parts
            else: # Start new block or after a paragraph break
                current_text_block = p_val.strip() 
    
    if current_text_block: # Add any remaining text
        final_parts.append(current_text_block)
        
    return [fp for fp in final_parts if fp.strip() or fp == "\n\n"]


def translate_text_in_chunks(text_to_translate, translator_pipe, tokenizer, max_chunk_tokens=400):
    if not text_to_translate or not text_to_translate.strip(): return ""
    if not translator_pipe or not tokenizer:
        print("WARNING: Translator pipe or tokenizer not available for chunked translation.")
        return bilingual("Translation service error.", "خطأ في خدمة الترجمة.")

    source_parts = []
    if NLTK_PUNKT_AVAILABLE:
        try:
            source_parts = nltk.sent_tokenize(text_to_translate)
            print(f"INFO: NLTK split text into {len(source_parts)} sentences for translation.")
            if not source_parts and text_to_translate.strip():
                raise ValueError("NLTK returned empty sentence list for non-empty text.")
        except Exception as e_nltk:
            print(f"WARNING: NLTK sentence tokenization error: {e_nltk}. Falling back to simple splitting.")
            source_parts = simple_splitter_by_paragraph_and_length(text_to_translate)
            print(f"INFO: Fallback splitter created {len(source_parts)} parts for translation.")
    else:
        print("INFO: NLTK 'punkt' was not available. Using simple_splitter for translation.")
        source_parts = simple_splitter_by_paragraph_and_length(text_to_translate)
        print(f"INFO: Fallback splitter created {len(source_parts)} parts for translation.")

    if not source_parts:
        print(f"ERROR: Text splitting failed to produce parts for translation: {text_to_translate[:200]}...")
        return bilingual("Translation splitting error.", "خطأ في تقسيم النص للترجمة.")

    translated_text_parts = []
    was_any_input_part_too_long_for_model = False 
    
    current_chunk_source_parts = []
    current_chunk_tokens_count = 0
    
    for i, part_text in enumerate(source_parts):
        # Handle paragraph markers from simple_splitter
        if part_text == "\n\n": 
            if current_chunk_source_parts: # Translate accumulated chunk before paragraph break
                chunk_to_translate = " ".join(current_chunk_source_parts)
                # print(f"DEBUG: Translating chunk before paragraph break (~{current_chunk_tokens_count} tokens)")
                try:
                    translated_chunk_list = translator_pipe(chunk_to_translate, max_new_tokens=int(current_chunk_tokens_count * 3.0 + 150)) # Generous output allowance
                    translated_text_parts.append(translated_chunk_list[0]['translation_text'])
                except Exception as e_chunk:
                    print(f"ERROR: --- ERROR TRANSLATING CHUNK (before paragraph) ---"); print(f"Failed chunk: '{chunk_to_translate}'"); traceback.print_exc()
                    translated_text_parts.append(f" [{bilingual('Chunk error P1', 'خطأ جزء ف1')}] ")
                current_chunk_source_parts = [] 
                current_chunk_tokens_count = 0
            translated_text_parts.append("\n\n") # Add paragraph break to translated parts
            continue
        
        if not part_text.strip(): continue # Skip empty parts

        part_tokens_ids = tokenizer.encode(part_text, add_special_tokens=False)
        
        if current_chunk_source_parts and (current_chunk_tokens_count + len(part_tokens_ids) > max_chunk_tokens):
            chunk_to_translate = " ".join(current_chunk_source_parts)
            # print(f"DEBUG: Translating chunk (~{current_chunk_tokens_count} input tokens): '{chunk_to_translate[:100]}...'")
            try:
                translated_chunk_list = translator_pipe(chunk_to_translate, max_new_tokens=int(current_chunk_tokens_count * 3.0 + 150))
                translated_text_parts.append(translated_chunk_list[0]['translation_text'])
            except Exception as e_chunk:
                print(f"ERROR: --- ERROR TRANSLATING CHUNK ---"); print(f"Failed chunk: '{chunk_to_translate}'"); traceback.print_exc()
                translated_text_parts.append(f" [{bilingual('Chunk error C1', 'خطأ جزء ج1')}] ")
            current_chunk_source_parts = [] 
            current_chunk_tokens_count = 0
        
        current_chunk_source_parts.append(part_text)
        current_chunk_tokens_count += len(part_tokens_ids)

        # tokenizer.model_max_length might be None for some tokenizers, so use a fallback.
        # Helsinki models are typically 512.
        model_max_input_len = tokenizer.model_max_length if tokenizer.model_max_length and tokenizer.model_max_length > 0 else 512
        if len(part_tokens_ids) > (model_max_input_len - 20) : 
            print(f"WARNING: A single part/sentence (index {i}) is very long ({len(part_tokens_ids)} tokens) and might be truncated by the translation model's input limit (approx {model_max_input_len}).")
            was_any_input_part_too_long_for_model = True

    if current_chunk_source_parts: # Translate any remaining part
        chunk_to_translate = " ".join(current_chunk_source_parts)
        # print(f"DEBUG: Translating FINAL chunk (~{current_chunk_tokens_count} input tokens): '{chunk_to_translate[:100]}...'")
        try:
            translated_chunk_list = translator_pipe(chunk_to_translate, max_new_tokens=int(current_chunk_tokens_count * 3.0 + 150))
            translated_text_parts.append(translated_chunk_list[0]['translation_text'])
        except Exception as e_chunk:
            print(f"ERROR: --- ERROR TRANSLATING FINAL CHUNK ---"); print(f"Failed chunk: '{chunk_to_translate}'"); traceback.print_exc()
            translated_text_parts.append(f" [{bilingual('Chunk error F1', 'خطأ جزء أخ1')}] ")
    
    final_translation_parts = []
    for p_text in translated_text_parts:
        if p_text == "\n\n":
            if not final_translation_parts or final_translation_parts[-1] != "\n\n": 
                final_translation_parts.append("\n\n")
        elif p_text.strip():
            final_translation_parts.append(p_text.strip())
    
    final_translation = ""
    for i, p_val in enumerate(final_translation_parts):
        if p_val == "\n\n":
            final_translation += "\n\n"
        else:
            final_translation += p_val
            if i < len(final_translation_parts) - 1 and final_translation_parts[i+1] != "\n\n" and not p_val.endswith("\n"):
                final_translation += " " 
    
    if was_any_input_part_too_long_for_model:
        final_translation += bilingual("\n\n(Note: Some parts of the English text were very long and may have been truncated by the translation model.)",
                                      "\n\n(ملاحظة: كانت بعض أجزاء النص الإنجليزي الأصلي طويلة جدًا وربما تم اختصارها بواسطة نموذج الترجمة.)")
    return final_translation.strip()

def analyze(files, ui_prompt_bilingual):
    print(f"\n--- ANALYZE FUNCTION CALLED (Final Version Candidate) ---")
    if not med_pipe: return [], bilingual("Error: MedGemma pipeline not loaded.", "خطأ: MedGemma غير مُحمّل."), None
    if not files: return [], bilingual("Error: No images uploaded.", "خطأ: لم تُرفع صور."), None
    
    imgs_pil = []
    try:
        for file_path_from_gradio in files:
            loaded_img = load_image_from_path(file_path_from_gradio)
            if loaded_img: imgs_pil.append(loaded_img)
    except Exception as e: return [], f"{bilingual('Error loading images', 'خطأ تحميل الصور')}: {e}", None
    if not imgs_pil: return [], bilingual("Error: Failed to load valid images.", "خطأ: فشل تحميل صور صالحة."), None
    
    target_image_pil = imgs_pil[0]
    if len(imgs_pil) > 1: print(f"INFO: Multiple images ({len(imgs_pil)}) uploaded, processing the first one.")

    actual_prompt_for_model = ui_prompt_bilingual 
    if not actual_prompt_for_model.strip(): actual_prompt_for_model = bilingual("Describe this radiological image.", "صف هذه الصورة الإشعاعية.")
    
    messages_for_pipeline = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
        {"role": "user", "content": [{"type": "text", "text": actual_prompt_for_model}, {"type": "image", "image": target_image_pil}]}
    ]
    en_report = bilingual("No English report generated.", "لم يتم إنشاء تقرير إنجليزي.")
    ar_report_text = bilingual("No Arabic report generated.", "لم يتم إنشاء تقرير عربي.")
    try:
        # print(f"DEBUG: Calling MedGemma pipeline on device: {med_pipe.device}...")
        with torch.no_grad():
            output_from_pipeline = med_pipe(text=messages_for_pipeline, max_new_tokens=2048) # Generous token limit for MedGemma
        
        if output_from_pipeline and isinstance(output_from_pipeline, list) and output_from_pipeline[0]:
            generated_chat_history = output_from_pipeline[0].get("generated_text")
            if generated_chat_history and isinstance(generated_chat_history, list) and generated_chat_history:
                last_message = generated_chat_history[-1]
                if isinstance(last_message, dict) and "content" in last_message: en_report = last_message["content"]
                elif isinstance(last_message, str): en_report = last_message
                else: en_report = str(generated_chat_history) 
            else: en_report = str(output_from_pipeline)
        else:
            return imgs_pil, bilingual("Error: MedGemma unexpected response.", "خطأ: استجابة MedGemma غير متوقعة."), None
        
        # Check if the report is still the default "No English report generated."
        no_en_report_indicator = bilingual("No English report generated.","لم يتم إنشاء تقرير إنجليزي.").split('/')[0].strip()
        if not en_report.strip() or en_report.strip() == no_en_report_indicator:
             en_report = bilingual("Assistant provided an empty or default response.", "رد المساعد فارغ/افتراضي.")
    except Exception as e:
        print(f"ERROR: RAW MedGemma Inference Error: {e}"); traceback.print_exc()
        return imgs_pil, f"{bilingual('Error MedGemma inference', 'خطأ استدلال MedGemma')}: {e}", None
            
    print(f"INFO: English report generated (length {len(en_report)} chars).")
    
    no_en_report_indicator_check = bilingual("No English report generated.","لم يتم إنشاء تقرير إنجليزي.").split('/')[0].strip()
    is_en_report_valid = en_report and en_report.strip() and en_report.strip() != no_en_report_indicator_check and not en_report.startswith(bilingual("Assistant empty/default response.",".").split('/')[0].strip())

    if mt_pipe and translation_tokenizer and is_en_report_valid:
        print("INFO: Starting chunked translation...")
        try:
            ar_report_text = translate_text_in_chunks(en_report, mt_pipe, translation_tokenizer, max_chunk_tokens=400) 
            print(f"INFO: Chunked Arabic translation generated (length {len(ar_report_text)} chars).")
        except Exception as e_trans_chunk: # Should be caught within translate_text_in_chunks
            print(f"ERROR: --- CATCH-ALL FOR CHUNKED TRANSLATION IN analyze() ---"); traceback.print_exc()
            ar_report_text = bilingual("Error during chunked translation process.", "خطأ أثناء عملية الترجمة المجزأة.")
    elif not mt_pipe or not translation_tokenizer:
        ar_report_text = bilingual("Translation service/tokenizer unavailable.", "خدمة الترجمة أو التوكنايزر غير متوفر.")
    elif not is_en_report_valid:
        print("INFO: English report was empty, default, or indicated an error; skipping translation.")
        ar_report_text = bilingual("Arabic translation skipped due to issue with English report.", "تم تخطي الترجمة العربية بسبب مشكلة في التقرير الإنجليزي.")
    
    ui_en_header = bilingual('English Report', 'التقرير بالإنجليزية')
    ui_ar_header = bilingual('Arabic Report', 'التقرير بالعربية')
    final_report_md = (
        f"📝 **{ui_en_header}:**\n\n{en_report}\n\n"
        f"---\n\n"
        f"📝 **{ui_ar_header}:**\n\n{ar_report_text}"
    )

    report_file_header_ar = "تقرير مساعد الأشعة MedGemma"
    report_file_header_en = "MedGemma Radiology Assistant Report"
    prompt_label_ar = "الاستفسار من الواجهة"
    prompt_label_en = "UI Prompt"
    en_report_header_ar = "--- التقرير باللغة الإنجليزية ---"
    en_report_header_en = "--- English Report ---"
    ar_report_header_ar = "--- التقرير باللغة العربية ---"
    ar_report_header_en = "--- Arabic Report ---"
    disclaimer_text_ar = "إخلاء مسؤولية: هذه الأداة مخصصة لأغراض البحث والعرض التوضيحي فقط. لا ينبغي استخدامها لاتخاذ قرارات تشخيص أو علاج طبي فعلي. استشر دائمًا أخصائيين طبيين مؤهلين."
    disclaimer_text_en = "Disclaimer: This tool is for research and demonstration purposes only. It should NOT be used for actual medical diagnosis or treatment decisions. Always consult with qualified medical professionals."
    report_file_content = f"""{report_file_header_ar}
{report_file_header_en}
====================================

{prompt_label_ar} / {prompt_label_en}:
{ui_prompt_bilingual}

{en_report_header_ar}
{en_report_header_en}
------------------------------------
{en_report}

{ar_report_header_ar}
{ar_report_header_en}
------------------------------------
{ar_report_text}

====================================
{disclaimer_text_ar}
{disclaimer_text_en}
"""
    report_path = "medgemma_dual_report.txt"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_file_content)
    except Exception as e_save: 
        print(f"ERROR: Error saving report: {e_save}"); report_path = None

    print(f"--- ANALYZE FUNCTION COMPLETED ---")
    return imgs_pil, final_report_md, report_path

EXAMPLE_IMAGE_PATH = "example_image.png" 
def get_interface():
    examples_list = None
    if os.path.exists(EXAMPLE_IMAGE_PATH):
        examples_list = [
            [[EXAMPLE_IMAGE_PATH], bilingual("Are there any signs of pneumonia?", "هل توجد أي علامات لالتهاب رئوي؟")],
        ]
    else: print(f"INFO: Example image '{EXAMPLE_IMAGE_PATH}' not found. Examples will be disabled.")

    app_greeting_ar = "مرحباً بك في مساعد الأشعة MedGemma"
    app_greeting_en = "Welcome to the MedGemma Radiology Assistant"
    app_short_description_ar = "حمّل صور الأشعة الطبية وأدخل استفسارًا للحصول على تحليل بالذكاء الاصطناعي باللغتين الإنجليزية والعربية."
    app_short_description_en = "Upload medical radiology images and enter a prompt for AI-driven analysis in both English and Arabic."
    supported_formats_text = "Supported file formats: JPG, PNG, DICOM (.dcm), NIfTI (.nii, .nii.gz)."
    disclaimer_ar_line1 = "⚠️ **إخلاء مسؤولية:** هذه الأداة مخصصة لأغراض البحث والعرض التوضيحي فقط."
    disclaimer_ar_line2 = "لا ينبغي استخدامها لاتخاذ قرارات تشخيص أو علاج طبي فعلي. استشر دائمًا أخصائيين طبيين مؤهلين."
    disclaimer_en_line1 = "⚠️ **Disclaimer:** This tool is for research and demonstration purposes only."
    disclaimer_en_line2 = "It should NOT be used for actual medical diagnosis or treatment decisions. Always consult with qualified medical professionals."
    full_description = f"""
<div style="text-align: center; margin-bottom: 20px;">
    <p style="font-size: 1.1em;">{app_greeting_ar}</p>
    <p style="font-size: 1.1em;">{app_greeting_en}</p>
</div>
<p>{app_short_description_ar}<br>{app_short_description_en}</p>
<p>{supported_formats_text}</p>
<hr style="margin-top: 20px; margin-bottom: 20px;">
<p style="font-size: 0.9em; color: #757575;">
    {disclaimer_ar_line1}<br>{disclaimer_ar_line2}<br><br>
    {disclaimer_en_line1}<br>{disclaimer_en_line2}
</p>
"""
    return gr.Interface(
        fn=analyze, 
        inputs=[
            gr.File(label=bilingual("Upload Images", "ارفع الصور"), 
                    file_types=[".jpg",".jpeg",".png",".dcm",".nii",".nii.gz"], 
                    file_count="multiple"),
            gr.Textbox(label=bilingual("Your Question / Prompt", "سؤالك / استفسارك"), 
                       lines=3,
                       placeholder=bilingual("e.g., What are the key findings? Are there any abnormalities?", 
                                             "مثال: ما هي النتائج الرئيسية؟ هل توجد أي تشوهات؟"),
                       value=bilingual("Describe the key findings in this radiological image. Provide details about any abnormalities, their location, size, and characteristics. What is the most likely diagnosis?", 
                                       "صف النتائج الرئيسية في هذه الصورة الإشعاعية. قدم تفاصيل حول أي تشوهات وموقعها وحجمها وخصائصها. ما هو التشخيص الأكثر احتمالا؟")),
        ],
        outputs=[ 
            gr.Gallery(label=bilingual("Uploaded Images", "الصور المعروضة"), columns=4, height="auto", preview=True),
            gr.Markdown(label=bilingual("AI Generated Report (English / Arabic)", "التقرير المُنشأ بواسطة الذكاء الاصطناعي (إنجليزي / عربي)"), elem_id="report_markdown"),
            gr.File(label=bilingual("Download Full Report", "تحميل التقرير الكامل"))
        ],
        title="🧠 MedGemma Radiology Assistant / مساعد الأشعة MedGemma", 
        description=full_description, 
        flagging_options=None, 
        examples=examples_list, 
        theme=gr.themes.Soft(),
        submit_btn=gr.Button(bilingual("Analyze / تحليل", "تحليل")), 
        clear_btn=gr.Button(bilingual("Clear Inputs / مسح المدخلات", "مسح المدخلات"))
    )

if __name__ == "__main__":
    critical_model_failed = False
    if not med_pipe: print("CRITICAL: MedGemma pipeline failed to load."); critical_model_failed = True
    if not mt_pipe or not translation_tokenizer: 
        print("WARNING: Translation model or its tokenizer failed to load. Arabic report might be unavailable or show errors.")
    
    interface = get_interface()
    if not critical_model_failed:
        print("INFO: Launching Gradio interface...")
        try: interface.launch(server_name="127.0.0.1", server_port=7860, share=False)
        except OSError as e:
            if "Cannot find empty port" in str(e) or "address already in use" in str(e).lower():
                print("WARNING: Port 7860 in use. Trying 7861..."); interface.launch(server_name="127.0.0.1", server_port=7861, share=False)
            else: print(f"ERROR: Failed to launch Gradio: {e}"); traceback.print_exc()
        except Exception as e: print(f"ERROR: Unexpected error launching Gradio: {e}"); traceback.print_exc()
    else: print("CRITICAL: App not launching due to critical model loading failures.")