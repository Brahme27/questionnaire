"""
Child Development Screening Tool
A comprehensive multi-modal Streamlit application for developmental assessment using Gemini AI
Supports: Likert scales, short answers, audio tasks, image-based questions, motor observations, yes/no, preferences
"""

import streamlit as st
import pandas as pd
import json
import base64
import io
import re
import requests
import urllib.parse
import random
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google import genai
from typing import Dict, List, Any, Optional, Tuple
from gtts import gTTS
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Child Development Screening Tool",
    page_icon="🧒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for healthcare color palette
st.markdown("""
<style>
    /* Question type badges */
    .question-type-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .badge-likert { background-color: #e3f2fd; color: #1565c0; }
    .badge-short-answer { background-color: #f3e5f5; color: #7b1fa2; }
    .badge-audio { background-color: #fff3e0; color: #ef6c00; }
    .badge-visual { background-color: #e8f5e9; color: #2e7d32; }
    .badge-motor { background-color: #fce4ec; color: #c2185b; }
    .badge-yesno { background-color: #e0f7fa; color: #00838f; }
    .badge-preference { background-color: #fff8e1; color: #ff8f00; }
    .badge-observation { background-color: #f1f8e9; color: #558b2f; }
    
    /* Motor task container */
    .motor-task {
        background-color: #fce4ec;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #c2185b;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# File paths for the data sources
DATA_FILES = {
    "Behavioral": "extracted_behavioral_questions.csv.xlsx",
    "Cognitive": "extracted_cognitive_questions.xlsx",
    "Learning Preferences": "extracted_learning_preferences_questions.csv.xlsx",
    "Sensory & Physical": "extracted_sensory_physical_questions.csv.xlsx"
}

# Question type patterns for classification
QUESTION_TYPE_PATTERNS = {
    "audio_instruction": [
        r"listen.*repeat",
        r"listen.*carefully",
        r"repeat.*sentence",
        r"follow.*instruction",
        r"say.*after",
        r"listen.*follow"
    ],
    "visual_identification": [
        r"look.*picture",
        r"identify.*shape",
        r"identify.*image",
        r"where.*hidden",
        r"point.*to",
        r"find.*in.*picture",
        r"what.*see.*image",
        r"match.*picture",
        r"identify.*color"
    ],
    "short_answer": [
        r"what did.*\?",
        r"what does.*\?",
        r"who.*\?",
        r"where.*\?",
        r"when.*\?",
        r"why.*\?",
        r"how many.*\?",
        r"what is.*\?",
        r"name.*\?"
    ],
    "motor_task": [
        r"can.*hop",
        r"can.*jump",
        r"can.*balance",
        r"can.*copy",
        r"can.*draw",
        r"can.*catch",
        r"can.*throw",
        r"can.*walk",
        r"can.*run",
        r"demonstrate.*ability",
        r"use.*hands.*manipulate",
        r"control.*coordination",
        r"kick.*ball",
        r"copy.*shape",
        r"geometric.*shape"
    ],
    "yes_no": [
        r"^can the child",
        r"^does the child",
        r"^is the child",
        r"^has the child"
    ],
    "preference": [
        r"prefer.*over",
        r"prefer.*hands-on",
        r"prefer.*listening",
        r"prefer.*reading",
        r"prefer.*visual",
        r"enjoy.*looking",
        r"enjoy.*listening",
        r"like.*better"
    ],
    "frequency_likert": [
        r"how often",
        r"how frequently"
    ]
}

# Response scales for different question types
LIKERT_SCALE = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3,
    "Always": 4
}

MOTOR_SKILL_SCALE = {
    "Cannot do at all": 0,
    "Needs significant help": 1,
    "Needs some help": 2,
    "Does with minor difficulty": 3,
    "Does independently": 4
}

YES_NO_SCALE = {
    "No": 0,
    "Sometimes": 1,
    "Yes": 2
}


def classify_question_type(question: str) -> str:
    """Classify question into appropriate type based on patterns."""
    question_lower = question.lower().strip()
    
    for q_type, patterns in QUESTION_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, question_lower):
                return q_type
    
    if "how often" in question_lower:
        return "frequency_likert"
    
    return "observation"


@st.cache_data
def load_questions(file_path: str) -> pd.DataFrame:
    """Load questions from Excel file."""
    try:
        df = pd.read_excel(file_path)
        return df
    except FileNotFoundError:
        st.error(f"❌ File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading {file_path}: {str(e)}")
        return pd.DataFrame()


def extract_tags(row: pd.Series, df_columns: List[str]) -> List[str]:
    """Extract tags from unnamed columns in the dataframe row."""
    tags = []
    tag_columns = [col for col in df_columns if col not in ['section_name', 'age', 'question']]
    
    for col in tag_columns:
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            tag = str(value).strip()
            tags.append(tag)
    
    return tags


def get_all_questions_from_file(df: pd.DataFrame, source_file: str = "") -> List[Dict[str, Any]]:
    """Get all valid questions with their classified type.
    
    Each question is tagged with its source file and a unique identifier
    to ensure questions strictly come from the given files and prevent duplicates.
    """
    if df.empty or 'question' not in df.columns:
        return []
    
    valid_df = df[df['question'].notna() & (df['question'].str.strip() != '')]
    
    if len(valid_df) == 0:
        return []
    
    questions_with_tags = []
    seen_questions = set()  # Track duplicates within the same file
    
    for idx, row in valid_df.iterrows():
        question_text = str(row['question']).strip()
        
        # Skip empty questions
        if not question_text:
            continue
        
        # Normalize for duplicate detection within file
        normalized = normalize_question_text(question_text)
        
        # Skip duplicates within the same source file
        if normalized in seen_questions:
            continue
        seen_questions.add(normalized)
        
        question_type = classify_question_type(question_text)
        
        question_data = {
            "index": int(idx),
            "question": question_text,
            "tags": extract_tags(row, df.columns.tolist()),
            "section": row.get('section_name', 'Unknown'),
            "question_type": question_type,
            "source_file": source_file,  # Track which file this question came from
            "original_row": int(idx)  # Track original row number for traceability
        }
        questions_with_tags.append(question_data)
    
    return questions_with_tags


def generate_audio_for_question(text: str) -> Optional[str]:
    """Generate audio using gTTS."""
    try:
        tts = gTTS(text=text, lang='en', slow=True)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        audio_base64 = base64.b64encode(audio_bytes.read()).decode()
        return audio_base64
    except Exception as e:
        return None


def generate_audio_content_for_child(api_key: str, question: str, child_age: str) -> dict:
    """Generate age-appropriate audio content (sentence to repeat, instruction to follow, etc.)."""
    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""You are creating audio content for a child developmental assessment.

**Question type:** "{question}"
**Child's age:** {child_age}

Generate age-appropriate content based on the question type:

1. If it's a "repeat the sentence" task:
   - For toddlers (1-3 years): Simple 3-4 word sentences like "The dog is big" or "I like apples"
   - For preschoolers (3-5 years): 5-7 word sentences like "The red ball bounced on the floor"
   - For school age (5+ years): 8-10 word sentences like "The little girl played with her puppy in the garden"

2. If it's a "follow instructions" task:
   - For toddlers: Simple 1-step instructions like "Clap your hands" or "Touch your nose"
   - For preschoolers: 2-step instructions like "Stand up and then sit down" or "Touch your head then your toes"
   - For school age: 3-step instructions like "Stand up, turn around once, and then clap twice"

3. If it's a listening comprehension task:
   - Generate a short story (2-3 sentences) appropriate for the age

Return JSON format ONLY:
{{
    "audio_text": "The exact text to be spoken aloud",
    "instruction_for_parent": "What to tell the child before playing",
    "expected_response": "What the child should do/say"
}}

Return ONLY the JSON, no other text."""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        response_text = response.text.strip()
        
        # Parse JSON
        if "```" in response_text:
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        return json.loads(response_text)
        
    except Exception as e:
        # Fallback content based on common patterns
        if "repeat" in question.lower():
            return {
                "audio_text": "The cat sat on the mat.",
                "instruction_for_parent": "Ask the child to listen carefully and repeat what they hear.",
                "expected_response": "The child should repeat: The cat sat on the mat."
            }
        elif "instruction" in question.lower() or "follow" in question.lower():
            return {
                "audio_text": "Please clap your hands two times.",
                "instruction_for_parent": "Ask the child to listen and do what they hear.",
                "expected_response": "The child should clap their hands twice."
            }
        else:
            return {
                "audio_text": "Hello, can you say hello back to me?",
                "instruction_for_parent": "Play the audio and observe the child's response.",
                "expected_response": "The child should respond appropriately."
            }


def generate_image_for_question(api_key: str, question: str) -> Optional[dict]:
    """Generate an actual image using Gemini's image generation capability."""
    errors = []
    
    try:
        client = genai.Client(api_key=api_key)
        
        # Create a child-appropriate image prompt
        image_prompt = f"""Create a simple, colorful, child-friendly cartoon illustration for a developmental assessment.

Question context: "{question}"

Requirements:
- Simple cartoon style with bright colors
- Clear, easy to understand for children aged 2-7 years
- Educational and engaging
- Safe and appropriate for young children
- Single clear subject or scene
- No text or words in the image"""
        
        # Method 1: Try Imagen 3.0 generate-002 (latest version)
        try:
            from google.genai import types
            
            response = client.models.generate_images(
                model="imagen-3.0-generate-002",
                prompt=image_prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="1:1",
                    safety_filter_level="BLOCK_ONLY_HIGH",
                    person_generation="DONT_ALLOW"
                )
            )
            
            if response.generated_images and len(response.generated_images) > 0:
                image_bytes = response.generated_images[0].image.image_bytes
                image_base64 = base64.b64encode(image_bytes).decode()
                
                desc_response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=f"For this question: '{question}', what should the child identify? Brief answer."
                )
                
                return {
                    "image_base64": image_base64,
                    "correct_answer": desc_response.text if desc_response else "Identify the item"
                }
        except Exception as e1:
            errors.append(f"Imagen-3.0-002: {str(e1)}")
        
        # Method 2: Try Gemini 2.0 Flash experimental with image output
        try:
            from google.genai import types
            
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=image_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                )
            )
            
            # Check for image in response parts
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        image_data = part.inline_data.data
                        if isinstance(image_data, bytes):
                            image_base64 = base64.b64encode(image_data).decode()
                        else:
                            image_base64 = image_data
                        
                        return {
                            "image_base64": image_base64,
                            "correct_answer": "Identify what you see in the image"
                        }
        except Exception as e2:
            errors.append(f"Gemini-2.0-flash-exp: {str(e2)}")
        
        # Method 3: Try imagegeneration model
        try:
            from google.genai import types
            
            response = client.models.generate_images(
                model="imagegeneration@006",
                prompt=image_prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="1:1"
                )
            )
            
            if response.generated_images and len(response.generated_images) > 0:
                image_bytes = response.generated_images[0].image.image_bytes
                image_base64 = base64.b64encode(image_bytes).decode()
                
                return {
                    "image_base64": image_base64,
                    "correct_answer": "Identify what you see in the image"
                }
        except Exception as e3:
            errors.append(f"imagegeneration@006: {str(e3)}")
        
        # Method 4: Use free Pollinations.ai API as fallback
        try:
            # Create a simpler prompt for the free API
            simple_prompt = f"simple colorful cartoon for children, {question}, educational, cute, bright colors, no text"
            encoded_prompt = urllib.parse.quote(simple_prompt)
            
            # Pollinations.ai free image API - smaller size for faster loading
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=300&height=300&nologo=true"
            
            # Download the image using requests
            response = requests.get(image_url, timeout=60)
            if response.status_code == 200:
                image_bytes = response.content
                image_base64 = base64.b64encode(image_bytes).decode()
                
                # Get description using Gemini
                client = genai.Client(api_key=api_key)
                desc_response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=f"For this question: '{question}', what should the child identify? Give a brief 5-10 word answer."
                )
                
                return {
                    "image_base64": image_base64,
                    "correct_answer": desc_response.text if desc_response else "Identify what you see"
                }
            else:
                errors.append(f"Pollinations.ai: HTTP {response.status_code}")
        except Exception as e4:
            errors.append(f"Pollinations.ai: {str(e4)}")
        
        # Log all errors for debugging
        if errors:
            st.warning(f"Image generation attempts: {'; '.join(errors)}")
        
        return None
        
    except Exception as e:
        # Fallback to text description if image generation fails
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"""Create a simple visual scenario description for: "{question}"
                Return JSON: {{"scene_description": "what to draw/show", "correct_answer": "what child should identify"}}
                Return ONLY JSON."""
            )
            return {"text_fallback": response.text, "error": str(e)}
        except:
            return None


def transcribe_audio_with_gemini(api_key: str, audio_bytes: bytes) -> str:
    """Transcribe audio using Gemini's multimodal capabilities."""
    try:
        client = genai.Client(api_key=api_key)
        
        # Convert audio bytes to base64
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        # Use Gemini to transcribe/analyze the audio
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                {
                    "parts": [
                        {"text": "Listen to this audio recording of a child responding to a developmental assessment. Transcribe what the child said and assess the clarity of their speech. Return in format: TRANSCRIPTION: [what was said] | CLARITY: [clear/partially clear/unclear]"},
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": audio_base64
                            }
                        }
                    ]
                }
            ]
        )
        return response.text
    except Exception as e:
        return f"Could not transcribe audio: {str(e)}"


def normalize_question_text(question: str) -> str:
    """Normalize question text for duplicate detection."""
    # Convert to lowercase, remove extra whitespace, strip punctuation for comparison
    normalized = question.lower().strip()
    # Remove multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    # Remove leading/trailing punctuation for comparison
    normalized = re.sub(r'^[^\w]+|[^\w]+$', '', normalized)
    return normalized


# Section-specific question counts (Behavioral needs more questions per therapist feedback)
SECTION_QUESTION_COUNTS = {
    "Behavioral": 12,  # More questions for behavioral assessment
    "Cognitive": 10,
    "Learning Preferences": 8,
    "Sensory & Physical": 10
}

def select_age_appropriate_questions_with_ai(api_key: str, child_age: str, all_questions: Dict[str, List[Dict]], num_questions: int = 10) -> Dict[str, List[Dict]]:
    """Use Gemini AI to select age-appropriate questions with variety in types.
    
    Ensures:
    - No duplicate questions across ANY sections
    - Questions come strictly from the source files
    - Each question has a unique identifier
    - RANDOMIZATION: Different questions each session
    - Section-specific question counts (Behavioral gets more questions)
    """
    try:
        client = genai.Client(api_key=api_key)
        
        selected_questions = {}
        # Global set to track used questions across ALL sections (normalized text for comparison)
        global_used_questions = set()
        # Also track by original index and section to ensure uniqueness
        global_used_indices = set()  # Format: "section_name:original_index"
        
        # Generate a random number for this selection session to ensure variety
        session_random = random.randint(1, 10000)
        
        for section_name, questions in all_questions.items():
            if not questions:
                selected_questions[section_name] = []
                continue
            
            # Filter out questions already used in previous sections
            available_questions = []
            for q in questions:
                normalized_text = normalize_question_text(q['question'])
                unique_id = f"{section_name}:{q['index']}"
                
                # Check both normalized text AND unique index to prevent duplicates
                if normalized_text not in global_used_questions and unique_id not in global_used_indices:
                    available_questions.append(q)
            
            if not available_questions:
                selected_questions[section_name] = []
                continue
            
            # RANDOMIZE: Shuffle available questions before presenting to AI
            random.shuffle(available_questions)
            
            # Use section-specific question count
            section_num_questions = SECTION_QUESTION_COUNTS.get(section_name, num_questions)
            
            questions_list = "\n".join([f"{i+1}. [{q['question_type'].upper()}] {q['question']}" for i, q in enumerate(available_questions)])
            questions_to_select = min(section_num_questions, len(available_questions))
            
            prompt = f"""You are a pediatric developmental specialist with expertise in child development milestones.

**Task:** Select exactly {questions_to_select} developmentally appropriate questions for a child aged **{child_age}**.

**IMPORTANT - RANDOMIZATION REQUIREMENT:**
This is session #{session_random}. To ensure variety across different assessments, please select a DIFFERENT combination of questions than you might typically choose. Prioritize variety and randomness while still maintaining age-appropriateness.

**Section:** {section_name}

**CRITICAL - Age-Based Selection Guidelines:**

For INFANTS (0-12 months):
- Focus on: reflexes, basic motor skills, eye tracking, response to sounds, social smile
- Avoid: complex language, abstract thinking, fine motor precision

For TODDLERS (1-3 years):
- Focus on: walking, simple words, basic shapes/colors, parallel play, simple instructions
- Avoid: reading comprehension, complex motor sequences, abstract concepts

For PRESCHOOLERS (3-5 years):
- Focus on: colors, shapes, counting, simple stories, gross motor (jumping, hopping), social play
- Avoid: complex reading, writing sentences, advanced reasoning

For EARLY SCHOOL AGE (5-7 years):
- Focus on: reading basics, writing, following multi-step instructions, team play, fine motor
- Avoid: questions requiring advanced literacy or abstract thinking

For SCHOOL AGE (7-12 years):
- Focus on: reading comprehension, writing, problem-solving, coordination, peer relationships
- Can include more complex cognitive and social questions

**Available Questions:**
{questions_list}

**Instructions:**
1. Select ONLY questions that are developmentally appropriate for age {child_age}
2. EXCLUDE questions that are too advanced or too simple for this age
3. Include variety in question types when available
4. **SELECT RANDOMLY** - Do NOT always pick the first few questions. Spread your selection across the entire list.
5. Return ONLY a JSON array of question numbers: [1, 5, 8, ...]

**Important:** A 2-year-old should NOT get reading comprehension questions. A 10-year-old should NOT get basic reflex questions. Match the developmental stage!

**CRITICAL:** You must ONLY return numbers from the list above. Do NOT invent new questions. Return ONLY question numbers that exist in the list. Pick from different parts of the list, not just the beginning.

Return JSON array only:"""
            
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            response_text = response.text.strip()
            
            try:
                if "```" in response_text:
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                response_text = response_text.strip()
                
                selected_indices = json.loads(response_text)
                
                # Shuffle the AI's selection to add another layer of randomness
                random.shuffle(selected_indices)
                
                selected = []
                selected_in_section = set()  # Track normalized text within this section
                
                for idx in selected_indices:
                    if 1 <= idx <= len(available_questions):
                        question_obj = available_questions[idx - 1]
                        normalized_text = normalize_question_text(question_obj['question'])
                        unique_id = f"{section_name}:{question_obj['index']}"
                        
                        # Triple check: not in section, not in global used, not by unique ID
                        if (normalized_text not in selected_in_section and 
                            normalized_text not in global_used_questions and
                            unique_id not in global_used_indices):
                            
                            # Add source tracking to the question object
                            question_obj['source_section'] = section_name
                            question_obj['unique_id'] = unique_id
                            
                            selected.append(question_obj)
                            selected_in_section.add(normalized_text)
                            global_used_questions.add(normalized_text)
                            global_used_indices.add(unique_id)
                        
                        if len(selected) >= questions_to_select:
                            break
                
                # Fill remaining slots if AI didn't select enough (still avoiding duplicates)
                if len(selected) < questions_to_select:
                    for q in available_questions:
                        normalized_text = normalize_question_text(q['question'])
                        unique_id = f"{section_name}:{q['index']}"
                        
                        if (normalized_text not in selected_in_section and 
                            normalized_text not in global_used_questions and
                            unique_id not in global_used_indices):
                            
                            q['source_section'] = section_name
                            q['unique_id'] = unique_id
                            
                            selected.append(q)
                            selected_in_section.add(normalized_text)
                            global_used_questions.add(normalized_text)
                            global_used_indices.add(unique_id)
                            
                            if len(selected) >= questions_to_select:
                                break
                
                selected_questions[section_name] = selected
                
            except (json.JSONDecodeError, ValueError):
                # Fallback: select questions randomly, still avoiding all duplicates
                selected = []
                # Shuffle for randomness in fallback
                fallback_pool = available_questions.copy()
                random.shuffle(fallback_pool)
                
                for q in fallback_pool:
                    normalized_text = normalize_question_text(q['question'])
                    unique_id = f"{section_name}:{q['index']}"
                    
                    if (normalized_text not in global_used_questions and
                        unique_id not in global_used_indices):
                        
                        q['source_section'] = section_name
                        q['unique_id'] = unique_id
                        
                        selected.append(q)
                        global_used_questions.add(normalized_text)
                        global_used_indices.add(unique_id)
                        
                        if len(selected) >= num_questions:
                            break
                selected_questions[section_name] = selected
        
        # Final validation: ensure no duplicates exist across all sections
        all_selected_texts = []
        for section, questions in selected_questions.items():
            for q in questions:
                normalized = normalize_question_text(q['question'])
                if normalized in all_selected_texts:
                    # This should never happen, but log if it does
                    st.warning(f"Duplicate detected and removed: {q['question'][:50]}...")
                    questions.remove(q)
                else:
                    all_selected_texts.append(normalized)
        
        # Shuffle final questions within each section for display randomness
        for section in selected_questions:
            random.shuffle(selected_questions[section])
        
        return selected_questions
        
    except Exception as e:
        st.error(f"Error selecting questions: {str(e)}")
        # Fallback with strict duplicate prevention and randomization
        selected_questions = {}
        global_used_questions = set()
        global_used_indices = set()
        
        for section, questions in all_questions.items():
            selected = []
            # Shuffle questions for random selection in fallback
            shuffled_questions = questions.copy()
            random.shuffle(shuffled_questions)
            
            # Use section-specific question count in fallback too
            section_num_questions = SECTION_QUESTION_COUNTS.get(section, num_questions)
            
            for q in shuffled_questions:
                normalized_text = normalize_question_text(q['question'])
                unique_id = f"{section}:{q['index']}"
                
                if (normalized_text not in global_used_questions and
                    unique_id not in global_used_indices):
                    
                    q['source_section'] = section
                    q['unique_id'] = unique_id
                    
                    selected.append(q)
                    global_used_questions.add(normalized_text)
                    global_used_indices.add(unique_id)
                    
                    if len(selected) >= section_num_questions:
                        break
            
            # Shuffle final selection order
            random.shuffle(selected)
            selected_questions[section] = selected
        return selected_questions


def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'questions_loaded': False,
        'all_questions': {},
        'all_questions_raw': {},
        'responses': {},
        'assessment_generated': False,
        'assessment_result': "",
        'child_age_for_questions': "",
        'audio_cache': {},
        'image_cache': {},
        'random_seed': int(time.time() * 1000) % (2**32)  # Unique seed per session
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_all_questions_raw():
    """Load ALL questions from all data files.
    
    Each question is tagged with its source to ensure questions
    strictly come from the given Excel files.
    Questions are shuffled to ensure randomization across sessions.
    """
    all_questions = {}
    global_seen_questions = set()  # Track questions across ALL files to prevent cross-file duplicates
    
    # Create a unique random seed based on current time and session
    random_seed = int(time.time() * 1000) % (2**32)
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = random_seed
    
    for section_name, file_path in DATA_FILES.items():
        df = load_questions(file_path)
        if not df.empty:
            questions = get_all_questions_from_file(df, source_file=file_path)
            
            # Filter out any questions that appeared in previous files
            unique_questions = []
            for q in questions:
                normalized = normalize_question_text(q['question'])
                if normalized not in global_seen_questions:
                    global_seen_questions.add(normalized)
                    unique_questions.append(q)
            
            # Shuffle questions for randomization using session-specific seed
            section_seed = st.session_state.random_seed + hash(section_name) % 1000
            random.seed(section_seed)
            random.shuffle(unique_questions)
            
            all_questions[section_name] = unique_questions
            
    return all_questions


def get_question_type_badge(q_type: str) -> str:
    """Return HTML badge for question type."""
    type_labels = {
        "frequency_likert": ("📊 Frequency Rating", "badge-likert"),
        "short_answer": ("✏️ Short Answer", "badge-short-answer"),
        "audio_instruction": ("🎧 Audio Task", "badge-audio"),
        "visual_identification": ("🖼️ Visual Task", "badge-visual"),
        "motor_task": ("🏃 Motor Skill", "badge-motor"),
        "yes_no": ("✅ Yes/No", "badge-yesno"),
        "preference": ("💡 Preference", "badge-preference"),
        "observation": ("👁️ Observation", "badge-observation")
    }
    label, badge_class = type_labels.get(q_type, ("📝 Question", "badge-observation"))
    return f'<span class="question-type-badge {badge_class}">{label}</span>'


# ============== QUESTION RENDERERS ==============

def render_likert_question(question: str, key: str) -> Dict:
    """Render a Likert scale question."""
    response = st.radio(
        "Your response",
        options=list(LIKERT_SCALE.keys()),
        index=2,  # Default to "Sometimes"
        key=key,
        horizontal=True,
        label_visibility="collapsed"
    )
    return {"response": response, "score": LIKERT_SCALE[response], "response_type": "likert"}


def render_short_answer_question(question: str, key: str) -> Dict:
    """Render a short answer question."""
    st.info("💡 **Comprehension Task**: Ask the child this question and record their response.")
    
    response = st.text_area(
        "Child's response",
        key=key,
        placeholder="Type what the child said...",
        height=100
    )
    
    # Rate the response quality
    quality = st.radio(
        "Response quality:",
        options=["No response", "Unclear/Off-topic", "Partially correct", "Correct"],
        key=f"{key}_quality",
        horizontal=True
    )
    
    score_map = {"No response": 0, "Unclear/Off-topic": 1, "Partially correct": 2, "Correct": 3}
    
    return {
        "response": response,
        "quality": quality,
        "score": score_map[quality],
        "response_type": "short_answer"
    }


def get_sentence_word_count_by_age(child_age: str) -> Tuple[int, int]:
    """Get the required word count range for repeat sentences based on child's age."""
    age_lower = child_age.lower()
    
    # Parse age to determine word count
    if any(x in age_lower for x in ['1 year', '12 month', '18 month']):
        return (2, 3)  # 1-2 years: 2-3 words
    elif any(x in age_lower for x in ['2 year', '24 month']):
        return (3, 4)  # 2-3 years: 3-4 words (as per therapist feedback)
    elif any(x in age_lower for x in ['3 year', '36 month']):
        return (3, 4)  # 3 years: 3-4 words
    elif any(x in age_lower for x in ['4 year', '48 month']):
        return (4, 5)  # 4 years: 4-5 words
    elif any(x in age_lower for x in ['5 year', '60 month']):
        return (5, 6)  # 5 years: 5-6 words
    elif any(x in age_lower for x in ['6 year', '72 month']):
        return (6, 7)  # 6 years: 6-7 words
    elif any(x in age_lower for x in ['7 year', '84 month']):
        return (7, 8)  # 7 years: 7-8 words
    else:
        # 8+ years
        return (8, 10)


def validate_sentence_word_count(sentence: str, min_words: int, max_words: int) -> bool:
    """Validate that a sentence has the correct word count."""
    words = sentence.strip().split()
    return min_words <= len(words) <= max_words


def generate_audio_content_for_child(api_key: str, question: str, child_age: str) -> Dict:
    """Generate age-appropriate audio content for the child based on the question type."""
    try:
        client = genai.Client(api_key=api_key)
        
        # Get strict word count requirements based on age
        min_words, max_words = get_sentence_word_count_by_age(child_age)
        
        # Parse the question to understand what type of audio task it is
        prompt = f"""You are generating age-appropriate audio content for a developmental screening assessment.

CHILD'S AGE: {child_age}

ORIGINAL QUESTION/TASK: {question}

Based on the question, generate specific audio content for the child. The content should be:
1. Age-appropriate for a {child_age} old child
2. Clear and easy to understand
3. Engaging but simple

Analyze the task type and generate appropriate content:

**CRITICAL - If it's a REPEAT/ECHO task** (child should repeat words/sentences):
⚠️ STRICT WORD COUNT REQUIREMENT: Generate EXACTLY {min_words} to {max_words} words for this age group.

Age-specific sentence requirements (MUST FOLLOW STRICTLY):
- For 1-2 year olds: EXACTLY 2-3 words (e.g., "Big red ball")
- For 2-3 year olds: EXACTLY 3-4 words (e.g., "The dog runs fast")
- For 3-4 year olds: EXACTLY 3-4 words (e.g., "Cat drinks milk now")
- For 4-5 year olds: EXACTLY 4-5 words (e.g., "The bird flies very high")
- For 5-6 year olds: EXACTLY 5-6 words (e.g., "The happy dog plays outside today")
- For 6-7 year olds: EXACTLY 6-7 words (e.g., "The little boy ate his lunch quickly")
- For 7+ year olds: EXACTLY 8-10 words

COUNT YOUR WORDS CAREFULLY! Do not exceed {max_words} words for this {child_age} old child.

**If it's a FOLLOW DIRECTIONS task** (child should do something):
- For under 3 years: 1 simple instruction (e.g., "Clap your hands")
- For 3-4 years: 1-2 step instructions (e.g., "Touch your nose, then your ears")
- For 5+ years: 2-3 step instructions

**If it's a LISTENING COMPREHENSION task**:
- Generate a short story appropriate for age, then a simple question

**If it's a SOUND/PHONICS task**:
- Generate age-appropriate sounds or words to identify

Respond in this exact JSON format:
{{
    "audio_text": "The EXACT text that will be spoken (for repeat tasks: MUST be {min_words}-{max_words} words)",
    "instruction_for_parent": "Brief instruction for parent/assessor about this task",
    "expected_response": "What a successful response looks like",
    "task_type": "repeat|directions|comprehension|phonics|rhythm|other",
    "word_count": <number of words in audio_text for repeat tasks>
}}

IMPORTANT: For repeat tasks, COUNT your words and ensure audio_text has EXACTLY {min_words} to {max_words} words!"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        response_text = response.text.strip()
        
        # Parse JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        import json
        content = json.loads(response_text)
        
        # Validate word count for repeat tasks
        if content.get('task_type') == 'repeat':
            audio_text = content.get('audio_text', '')
            if not validate_sentence_word_count(audio_text, min_words, max_words):
                # Generate a fallback sentence with correct word count
                fallback_sentences = {
                    (2, 3): ["Big red ball", "Cat is soft", "Dog runs fast"],
                    (3, 4): ["The dog is big", "I like apples", "Bird flies high"],
                    (4, 5): ["The cat is sleeping now", "I see blue sky", "The ball bounces high"],
                    (5, 6): ["The happy dog plays outside", "I like to eat apples", "The red bird sings songs"],
                    (6, 7): ["The little girl plays with dolls", "My cat likes to drink milk", "The boy runs to the park"],
                    (7, 8): ["The brown dog runs fast in the park", "My little sister likes to play games"],
                    (8, 10): ["The happy children played with their toys in the sunny garden", "After breakfast the boy went outside to play with friends"]
                }
                # Get appropriate fallback
                fallback_list = fallback_sentences.get((min_words, max_words), ["Hello there"])
                content['audio_text'] = random.choice(fallback_list)
                content['word_count'] = len(content['audio_text'].split())
        
        return content
        
    except Exception as e:
        st.warning(f"Could not generate specific content: {e}")
        # Fallback to simple content with correct word count
        min_words, max_words = get_sentence_word_count_by_age(child_age)
        fallback_sentences = {
            (2, 3): "Big red ball",
            (3, 4): "The dog runs",
            (4, 5): "The cat is soft",
            (5, 6): "The bird flies very high",
            (6, 7): "The happy dog plays with ball",
            (7, 8): "The little boy ate his lunch fast",
            (8, 10): "The children played happily in the sunny garden today"
        }
        fallback_text = fallback_sentences.get((min_words, max_words), "Hello there")
        
        return {
            "audio_text": fallback_text,
            "instruction_for_parent": "Play for child and observe response",
            "expected_response": "Child attempts to repeat the sentence",
            "task_type": "repeat",
            "word_count": len(fallback_text.split())
        }


def render_audio_instruction_question(question: str, key: str, api_key: str) -> Dict:
    """Render an audio instruction question with playback and child recording."""
    
    st.info("🎧 **Audio Task**: Play the audio for the child to listen, then record their response.")
    
    # Generate age-appropriate audio content (not just the question text)
    child_age = st.session_state.get('child_age_for_questions', '4 years')
    
    if key not in st.session_state.audio_cache:
        with st.spinner("🤖 Generating age-appropriate audio content..."):
            # Generate specific content for the child
            audio_content = generate_audio_content_for_child(api_key, question, child_age)
            
            if audio_content:
                # Generate the actual audio from the specific text
                audio_text = audio_content.get("audio_text", question)
                audio_base64 = generate_audio_for_question(audio_text)
                
                if audio_base64:
                    st.session_state.audio_cache[key] = {
                        "audio_base64": audio_base64,
                        "audio_text": audio_text,
                        "instruction_for_parent": audio_content.get("instruction_for_parent", "Play for child"),
                        "expected_response": audio_content.get("expected_response", "Listen for child's response")
                    }
    
    # Display the audio task
    if key in st.session_state.audio_cache:
        cached = st.session_state.audio_cache[key]
        
        # Show parent instruction
        st.write(f"**📋 Instructions:** {cached.get('instruction_for_parent', 'Play the audio for the child.')}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**🔊 Step 1: Play this for the child:**")
            
            # Show what will be played
            with st.expander("👁️ See what the child will hear"):
                st.info(f"*\"{cached.get('audio_text', '')}\"*")
                st.caption(f"Expected response: {cached.get('expected_response', '')}")
            
            # Audio player
            audio_html = f'''
            <audio controls style="width: 100%;">
                <source src="data:audio/mp3;base64,{cached['audio_base64']}" type="audio/mp3">
            </audio>
            '''
            st.markdown(audio_html, unsafe_allow_html=True)
        
        with col2:
            st.write("**🎤 Step 2: Record child's response:**")
            recorded_audio = st.audio_input(
                "Click to record",
                key=f"{key}_recording"
            )
    else:
        st.warning(f"⚠️ Could not generate audio. Read aloud: *{question}*")
        col1, col2 = st.columns([1, 1])
        with col2:
            recorded_audio = st.audio_input(
                "Record child's response",
                key=f"{key}_recording"
            )
    
    # Process recorded audio
    transcription = ""
    
    if recorded_audio:
        st.success("✅ Audio recorded!")
        st.audio(recorded_audio, format="audio/wav")
        
        with st.spinner("🤖 Analyzing child's response..."):
            audio_bytes = recorded_audio.getvalue()
            
            # Get the expected text for comparison
            expected_text = ""
            if key in st.session_state.audio_cache:
                expected_text = st.session_state.audio_cache[key].get('audio_text', '')
            
            # Transcribe with context
            transcription = transcribe_audio_with_gemini(api_key, audio_bytes)
            
            if transcription and "Could not" not in transcription:
                st.write("**📝 What we heard:**")
                st.info(transcription)
    
    # Rating
    st.write("**📊 Rate the response:**")
    accuracy = st.radio(
        "How well did the child respond?",
        options=["No response", "Partial/Unclear", "Mostly accurate", "Fully accurate"],
        index=2 if recorded_audio else 0,  # Default based on whether audio was recorded
        key=f"{key}_rating",
        horizontal=True
    )
    
    score_map = {"No response": 0, "Partial/Unclear": 1, "Mostly accurate": 2, "Fully accurate": 3}
    
    # Get the actual audio text that was played
    audio_text_played = ""
    if key in st.session_state.audio_cache:
        audio_text_played = st.session_state.audio_cache[key].get('audio_text', '')
    
    return {
        "response": accuracy,
        "score": score_map[accuracy],
        "transcription": transcription,
        "audio_played": audio_text_played,
        "has_recording": recorded_audio is not None,
        "response_type": "audio_instruction"
    }


def render_visual_question(question: str, key: str, api_key: str) -> Dict:
    """Render a visual identification question with generated image."""
    
    st.info("🖼️ **Visual Task**: Show the image to the child and ask them to identify what they see.")
    
    # Generate image
    if key not in st.session_state.image_cache:
        with st.spinner("🎨 Generating image for this question... (this may take a moment)"):
            try:
                image_data = generate_image_for_question(api_key, question)
                if image_data:
                    st.session_state.image_cache[key] = image_data
                else:
                    # Store a marker that we tried but failed
                    st.session_state.image_cache[key] = {"generation_failed": True}
            except Exception as gen_err:
                st.session_state.image_cache[key] = {"generation_failed": True, "error": str(gen_err)}
    
    # Create two columns - image on left, question/response on right
    col_image, col_response = st.columns([1, 1])
    
    # Display the image in left column
    with col_image:
        if key in st.session_state.image_cache:
            cached_data = st.session_state.image_cache[key]
            
            if cached_data.get("generation_failed"):
                # Show fallback UI
                st.warning("⚠️ Image generation unavailable.")
                st.write(f"**Task:** {question}")
                st.caption("Present appropriate visual materials to the child.")
            
            elif "image_base64" in cached_data:
                # Display actual generated image - compact size
                st.write("**🖼️ Show to child:**")
                
                try:
                    # Display using st.image with fixed width
                    image_bytes = base64.b64decode(cached_data["image_base64"])
                    image = Image.open(io.BytesIO(image_bytes))
                    # Resize image if too large (max 300px width for display)
                    max_display_width = 280
                    if image.width > max_display_width:
                        ratio = max_display_width / image.width
                        new_size = (max_display_width, int(image.height * ratio))
                        image = image.resize(new_size, Image.Resampling.LANCZOS)
                    st.image(image, width='content')
                except Exception as img_err:
                    # Fallback to HTML img tag with size constraint
                    try:
                        image_html = f'<img src="data:image/png;base64,{cached_data["image_base64"]}" style="max-width: 280px; max-height: 280px; border-radius: 8px; border: 2px solid #4a9b8c;">'
                        st.markdown(image_html, unsafe_allow_html=True)
                    except:
                        st.error(f"Could not display image")
                
                # Show what to look for
                st.caption(f"✓ {cached_data.get('correct_answer', 'Ask what they see')[:100]}")
            
            elif "text_fallback" in cached_data:
                # Fallback to text description
                st.warning("⚠️ Using description instead.")
                try:
                    scene_json = cached_data["text_fallback"]
                    if "```" in scene_json:
                        scene_json = scene_json.split("```")[1]
                        if scene_json.startswith("json"):
                            scene_json = scene_json[4:]
                    scene_info = json.loads(scene_json.strip())
                    st.write("**Draw/show:**", scene_info.get("scene_description", "")[:150])
                except:
                    st.caption("Present visual materials for this question.")
        else:
            st.warning("Could not generate image.")
    
    # Child's response in right column
    with col_response:
        st.write("**👶 Child's Response:**")
        response = st.radio(
            "What did the child identify?",
            options=["Correct identification", "Partially correct", "Incorrect", "No response"],
            key=key,
            horizontal=False
        )
        
        details = st.text_input(
            "What did the child say/point to:",
            key=f"{key}_details",
            placeholder="e.g., 'Pointed to the red ball'"
        )
    
    score_map = {"Correct identification": 3, "Partially correct": 2, "Incorrect": 1, "No response": 0}
    
    return {
        "response": response,
        "score": score_map[response],
        "details": details,
        "response_type": "visual_identification"
    }


def render_motor_task_question(question: str, key: str) -> Dict:
    """Render a motor task observation question."""
    
    st.markdown(f"""
    <div class="motor-task">
        <strong>🏃 Motor Task to Observe:</strong><br>{question}
    </div>
    """, unsafe_allow_html=True)
    
    st.info("👀 Observe the child performing this task and rate their ability.")
    
    response = st.radio(
        "Skill level observed",
        options=list(MOTOR_SKILL_SCALE.keys()),
        index=3,  # Default to "Does with minor difficulty"
        key=key,
        horizontal=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        attempts = st.number_input("Attempts needed", min_value=1, max_value=10, value=1, key=f"{key}_attempts")
    with col2:
        consistency = st.radio(
            "Consistency",
            options=["Inconsistent", "Variable", "Mostly consistent", "Very consistent"],
            index=2,  # Default to "Mostly consistent"
            key=f"{key}_consistency",
            horizontal=True
        )
    
    notes = st.text_input(
        "Observations (optional)",
        key=f"{key}_notes",
        placeholder="e.g., 'Good balance, slight wobble on left foot'"
    )
    
    return {
        "response": response,
        "score": MOTOR_SKILL_SCALE[response],
        "attempts": attempts,
        "consistency": consistency,
        "notes": notes,
        "response_type": "motor_task"
    }


def render_yes_no_question(question: str, key: str) -> Dict:
    """Render a yes/no question."""
    
    response = st.radio(
        "Response",
        options=list(YES_NO_SCALE.keys()),
        key=key,
        horizontal=True
    )
    
    details = ""
    if response == "Sometimes":
        details = st.text_input(
            "Please specify",
            key=f"{key}_details",
            placeholder="When or how often?"
        )
    
    return {
        "response": response,
        "score": YES_NO_SCALE[response],
        "details": details,
        "response_type": "yes_no"
    }


def render_preference_question(question: str, key: str) -> Dict:
    """Render a preference-based question."""
    
    # Detect learning modality from question
    question_lower = question.lower()
    learning_modality = "General"
    if any(word in question_lower for word in ['listen', 'auditory', 'hear', 'sound', 'verbal', 'spoken']):
        learning_modality = "Auditory"
    elif any(word in question_lower for word in ['visual', 'picture', 'see', 'look', 'watch', 'image', 'diagram']):
        learning_modality = "Visual"
    elif any(word in question_lower for word in ['hands-on', 'touch', 'move', 'kinesthetic', 'physical', 'activity', 'do']):
        learning_modality = "Kinesthetic"
    elif any(word in question_lower for word in ['read', 'write', 'text', 'book', 'reading', 'writing']):
        learning_modality = "Reading/Writing"
    
    modality_colors = {
        "Auditory": "#e3f2fd",
        "Visual": "#e8f5e9", 
        "Kinesthetic": "#fff3e0",
        "Reading/Writing": "#f3e5f5",
        "General": "#f5f5f5"
    }
    
    st.markdown(f"""
    <div style="background-color: {modality_colors.get(learning_modality, '#f5f5f5')}; 
                padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 0.5rem;
                display: inline-block;">
        <span style="font-weight: 600;">🎯 Learning Style: {learning_modality}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 **Learning Preference**: Rate how strongly this describes the child.")
    
    response = st.radio(
        "How well does this describe the child?",
        options=["Not at all", "A little", "Somewhat", "Quite a bit", "Very much"],
        index=2,  # Default to "Somewhat"
        key=key,
        horizontal=True
    )
    
    score_map = {"Not at all": 0, "A little": 1, "Somewhat": 2, "Quite a bit": 3, "Very much": 4}
    
    return {
        "response": response,
        "score": score_map[response],
        "response_type": "preference",
        "learning_modality": learning_modality
    }


def render_observation_question(question: str, key: str) -> Dict:
    """Render a general observation question."""
    
    response = st.radio(
        "Your observation",
        options=list(LIKERT_SCALE.keys()),
        index=2,  # Default to "Sometimes"
        key=key,
        horizontal=True,
        label_visibility="collapsed"
    )
    
    return {
        "response": response,
        "score": LIKERT_SCALE[response],
        "response_type": "observation"
    }


def render_question(q_data: Dict, index: int, section_key: str, api_key: str) -> Dict:
    """Render a question based on its type."""
    question = q_data['question']
    q_type = q_data['question_type']
    tags = q_data['tags']
    key = f"{section_key}_q{index}"
    
    st.markdown(get_question_type_badge(q_type), unsafe_allow_html=True)
    st.markdown(f"**Question {index + 1}:** {question}")
    
    renderers = {
        "frequency_likert": lambda: render_likert_question(question, key),
        "short_answer": lambda: render_short_answer_question(question, key),
        "audio_instruction": lambda: render_audio_instruction_question(question, key, api_key),
        "visual_identification": lambda: render_visual_question(question, key, api_key),
        "motor_task": lambda: render_motor_task_question(question, key),
        "yes_no": lambda: render_yes_no_question(question, key),
        "preference": lambda: render_preference_question(question, key),
        "observation": lambda: render_observation_question(question, key)
    }
    
    result = renderers.get(q_type, lambda: render_observation_question(question, key))()
    
    result["question"] = question
    result["question_type"] = q_type
    result["tags"] = tags
    
    return result


def render_question_section(section_name: str, questions: List[Dict], section_key: str, api_key: str):
    """Render questions for a specific section."""
    st.subheader(f"📝 {section_name} Assessment")
    
    # Show question type summary
    type_counts = {}
    for q in questions:
        q_type = q.get('question_type', 'observation')
        type_counts[q_type] = type_counts.get(q_type, 0) + 1
    
    if type_counts:
        type_summary = " • ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in type_counts.items()])
        st.caption(f"📋 {type_summary}")
    
    st.markdown("---")
    
    if not questions:
        st.warning(f"No questions available for {section_name}.")
        return []
    
    responses = []
    for i, q_data in enumerate(questions):
        with st.container():
            response = render_question(q_data, i, section_key, api_key)
            responses.append(response)
            st.markdown("---")
    
    return responses


def check_all_questions_answered(all_responses: Dict[str, List[Dict]]) -> Tuple[bool, int, int, List[str]]:
    """Check if all questions have been answered across all sections.
    
    Returns: (all_answered, answered_count, total_count, missing_sections)
    """
    total_questions = 0
    answered_questions = 0
    missing_sections = []
    
    for section_name, responses in all_responses.items():
        section_total = len(responses)
        section_answered = 0
        
        for resp in responses:
            total_questions += 1
            response_value = resp.get('response')
            
            # Check if question has a valid response
            if response_value is not None and str(response_value).strip() != '':
                # For text responses, check they're not empty
                if resp.get('response_type') == 'short_answer':
                    if resp.get('quality') and resp.get('quality') != 'No response':
                        section_answered += 1
                        answered_questions += 1
                elif resp.get('response_type') == 'audio_instruction':
                    if resp.get('response') and resp.get('response') != 'No response':
                        section_answered += 1
                        answered_questions += 1
                else:
                    section_answered += 1
                    answered_questions += 1
        
        if section_answered < section_total:
            missing_sections.append(f"{section_name} ({section_answered}/{section_total})")
    
    all_answered = answered_questions >= total_questions and total_questions > 0
    return all_answered, answered_questions, total_questions, missing_sections


def calculate_domain_scores(responses: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Calculate comprehensive scores for each domain based on actual responses."""
    domain_scores = {}
    
    for section, section_responses in responses.items():
        if not section_responses:
            continue
            
        scores = []
        max_possible = 0
        question_details = []
        
        for resp in section_responses:
            score = resp.get('score', 0)
            q_type = resp.get('question_type', 'observation')
            
            # Determine max score based on question type
            if q_type in ['frequency_likert', 'preference', 'observation']:
                max_score = 4
            elif q_type in ['short_answer', 'audio_instruction', 'visual_identification']:
                max_score = 3
            elif q_type == 'motor_task':
                max_score = 4
            elif q_type == 'yes_no':
                max_score = 2
            else:
                max_score = 4
            
            scores.append(score)
            max_possible += max_score
            
            question_details.append({
                'question': resp.get('question', '')[:50],
                'score': score,
                'max_score': max_score,
                'type': q_type,
                'response': resp.get('response', ''),
                'tags': resp.get('tags', [])
            })
        
        if scores:
            total_score = sum(scores)
            avg_score = total_score / len(scores) if scores else 0
            percentage = (total_score / max_possible * 100) if max_possible > 0 else 0
            
            # Determine status based on percentage (for frequency-based concerns, lower is better)
            # For developmental skills, higher is better
            domain_scores[section] = {
                'total_score': total_score,
                'max_possible': max_possible,
                'average': round(avg_score, 2),
                'percentage': round(percentage, 1),
                'question_count': len(scores),
                'high_scores': sum(1 for s in scores if s >= 3),
                'low_scores': sum(1 for s in scores if s <= 1),
                'details': question_details
            }
    
    return domain_scores


def analyze_clinical_tags(responses: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Analyze clinical indicator tags from responses."""
    tag_analysis = {}
    
    for section, section_responses in responses.items():
        for resp in section_responses:
            for tag in resp.get('tags', []):
                tag_clean = str(tag).strip().upper()
                if tag_clean and len(tag_clean) > 1:
                    if tag_clean not in tag_analysis:
                        tag_analysis[tag_clean] = {
                            'high_scores': 0,
                            'total': 0,
                            'questions': [],
                            'sections': set()
                        }
                    tag_analysis[tag_clean]['total'] += 1
                    tag_analysis[tag_clean]['sections'].add(section)
                    
                    if resp.get('score', 0) >= 3:
                        tag_analysis[tag_clean]['high_scores'] += 1
                        tag_analysis[tag_clean]['questions'].append({
                            'question': resp.get('question', '')[:60],
                            'score': resp.get('score', 0),
                            'response': resp.get('response', '')
                        })
    
    # Convert sets to lists for JSON serialization
    for tag in tag_analysis:
        tag_analysis[tag]['sections'] = list(tag_analysis[tag]['sections'])
    
    return tag_analysis


def create_domain_radar_chart(domain_scores: Dict[str, Dict]) -> go.Figure:
    """Create a radar chart showing domain performance."""
    categories = list(domain_scores.keys())
    values = [domain_scores[cat]['percentage'] for cat in categories]
    
    # Close the radar chart
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(99, 179, 237, 0.3)',
        line=dict(color='rgb(49, 130, 206)', width=2),
        name='Score %'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10),
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        showlegend=False,
        title=dict(
            text="Domain Performance Overview",
            font=dict(size=16),
            x=0.5
        ),
        height=400,
        margin=dict(l=80, r=80, t=60, b=40)
    )
    
    return fig


def create_domain_bar_chart(domain_scores: Dict[str, Dict]) -> go.Figure:
    """Create a horizontal bar chart for domain scores."""
    domains = list(domain_scores.keys())
    percentages = [domain_scores[d]['percentage'] for d in domains]
    
    # Color based on score level
    colors = []
    for p in percentages:
        if p >= 75:
            colors.append('#48bb78')  # Green - typical/good
        elif p >= 50:
            colors.append('#ecc94b')  # Yellow - moderate
        elif p >= 25:
            colors.append('#ed8936')  # Orange - concern
        else:
            colors.append('#f56565')  # Red - significant concern
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=domains,
        x=percentages,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{p:.0f}%' for p in percentages],
        textposition='outside',
        textfont=dict(size=12)
    ))
    
    fig.update_layout(
        title=dict(
            text="Domain Scores Summary",
            font=dict(size=16),
            x=0.5
        ),
        xaxis=dict(
            title="Score Percentage",
            range=[0, 110],
            ticksuffix='%'
        ),
        yaxis=dict(title=""),
        height=300,
        margin=dict(l=150, r=60, t=60, b=40),
        showlegend=False
    )
    
    return fig


def create_question_type_pie_chart(responses: Dict[str, List[Dict]]) -> go.Figure:
    """Create a pie chart showing distribution of question types."""
    type_counts = {}
    
    for section, section_responses in responses.items():
        for resp in section_responses:
            q_type = resp.get('question_type', 'observation')
            type_label = q_type.replace('_', ' ').title()
            type_counts[type_label] = type_counts.get(type_label, 0) + 1
    
    labels = list(type_counts.keys())
    values = list(type_counts.values())
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#fa709a']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors[:len(labels)]),
        textinfo='label+percent',
        textfont=dict(size=11)
    )])
    
    fig.update_layout(
        title=dict(
            text="Assessment Types Used",
            font=dict(size=16),
            x=0.5
        ),
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False
    )
    
    return fig


def analyze_learning_preferences(responses: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Analyze learning preferences to identify learning modalities (Auditory, Visual, Kinesthetic, etc.)."""
    modality_scores = {
        "Auditory": {"total_score": 0, "max_score": 0, "count": 0, "keywords": ['listen', 'auditory', 'hear', 'sound', 'verbal', 'spoken', 'music', 'rhythm']},
        "Visual": {"total_score": 0, "max_score": 0, "count": 0, "keywords": ['visual', 'picture', 'see', 'look', 'watch', 'image', 'diagram', 'color', 'read']},
        "Kinesthetic": {"total_score": 0, "max_score": 0, "count": 0, "keywords": ['hands-on', 'touch', 'move', 'kinesthetic', 'physical', 'activity', 'do', 'manipulate', 'build']},
        "Reading/Writing": {"total_score": 0, "max_score": 0, "count": 0, "keywords": ['read', 'write', 'text', 'book', 'story', 'letter', 'word']}
    }
    
    # Get Learning Preferences section
    learning_responses = responses.get("Learning Preferences", [])
    
    for resp in learning_responses:
        question = resp.get('question', '').lower()
        score = resp.get('score', 0)
        
        # Determine which modality this question relates to
        for modality, data in modality_scores.items():
            if any(keyword in question for keyword in data['keywords']):
                data['total_score'] += score
                data['max_score'] += 4  # Max score for preference questions
                data['count'] += 1
                break
        else:
            # If no specific modality found, check if response has learning_modality
            if resp.get('learning_modality') and resp.get('learning_modality') != "General":
                modality = resp.get('learning_modality')
                if modality in modality_scores:
                    modality_scores[modality]['total_score'] += score
                    modality_scores[modality]['max_score'] += 4
                    modality_scores[modality]['count'] += 1
    
    # Calculate percentages
    result = {}
    for modality, data in modality_scores.items():
        if data['count'] > 0:
            percentage = (data['total_score'] / data['max_score'] * 100) if data['max_score'] > 0 else 0
            result[modality] = {
                'percentage': round(percentage, 1),
                'score': data['total_score'],
                'max_score': data['max_score'],
                'count': data['count']
            }
    
    return result


def create_learning_preferences_chart(responses: Dict[str, List[Dict]]) -> go.Figure:
    """Create a chart showing learning style preferences breakdown."""
    modality_data = analyze_learning_preferences(responses)
    
    if not modality_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No learning preference data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#718096")
        )
        fig.update_layout(
            height=250,
            title=dict(text="Learning Style Preferences", font=dict(size=16), x=0.5)
        )
        return fig
    
    modalities = list(modality_data.keys())
    percentages = [modality_data[m]['percentage'] for m in modalities]
    counts = [modality_data[m]['count'] for m in modalities]
    
    # Colors for each modality
    modality_colors = {
        "Auditory": "#3182ce",  # Blue
        "Visual": "#38a169",    # Green
        "Kinesthetic": "#dd6b20", # Orange
        "Reading/Writing": "#805ad5"  # Purple
    }
    colors = [modality_colors.get(m, "#718096") for m in modalities]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=modalities,
        y=percentages,
        marker_color=colors,
        text=[f'{p:.0f}%' for p in percentages],
        textposition='outside',
        textfont=dict(size=12, weight='bold'),
        hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}%<br>Items: %{customdata}<extra></extra>',
        customdata=counts
    ))
    
    fig.update_layout(
        title=dict(
            text="Learning Style Preferences",
            font=dict(size=16),
            x=0.5
        ),
        xaxis=dict(title="Learning Modality"),
        yaxis=dict(title="Preference Score", range=[0, 110], ticksuffix='%'),
        height=300,
        margin=dict(l=60, r=40, t=60, b=60),
        showlegend=False
    )
    
    # Add icons/emojis as annotations
    icons = {"Auditory": "🎧", "Visual": "👁️", "Kinesthetic": "🤸", "Reading/Writing": "📖"}
    for i, modality in enumerate(modalities):
        fig.add_annotation(
            x=modality,
            y=percentages[i] + 8,
            text=icons.get(modality, "📊"),
            showarrow=False,
            font=dict(size=16)
        )
    
    return fig


def create_clinical_flags_chart(tag_analysis: Dict[str, Dict]) -> go.Figure:
    """Create a chart showing clinical indicator flags with percentage-based comparison."""
    # Filter for significant tags
    significant_tags = {
        tag: data for tag, data in tag_analysis.items()
        if data['total'] >= 2 and data['high_scores'] >= 1
    }
    
    if not significant_tags:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No significant clinical flags identified",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#38a169")
        )
        fig.update_layout(
            height=200,
            title=dict(text="Clinical Indicator Analysis", font=dict(size=16), x=0.5)
        )
        return fig
    
    tags = list(significant_tags.keys())
    high_counts = [significant_tags[t]['high_scores'] for t in tags]
    total_counts = [significant_tags[t]['total'] for t in tags]
    # Calculate percentages for better comparison
    percentages = [(h / t * 100) if t > 0 else 0 for h, t in zip(high_counts, total_counts)]
    
    # Color based on severity percentage
    colors = []
    for p in percentages:
        if p >= 75:
            colors.append('#c53030')  # Dark red - significant
        elif p >= 50:
            colors.append('#f56565')  # Light red - moderate
        elif p >= 25:
            colors.append('#ed8936')  # Orange - mild concern
        else:
            colors.append('#ecc94b')  # Yellow - monitor
    
    fig = go.Figure()
    
    # Use grouped bar chart for clearer comparison
    fig.add_trace(go.Bar(
        name='High Score %',
        x=tags,
        y=percentages,
        marker_color=colors,
        text=[f'{p:.0f}%<br>({h}/{t})' for p, h, t in zip(percentages, high_counts, total_counts)],
        textposition='outside',
        textfont=dict(size=11)
    ))
    
    # Add threshold line at 50%
    fig.add_hline(y=50, line_dash="dash", line_color="#718096", 
                  annotation_text="Concern Threshold (50%)", 
                  annotation_position="top right")
    
    fig.update_layout(
        title=dict(
            text="Clinical Indicator Analysis (% of High Scores)",
            font=dict(size=16),
            x=0.5
        ),
        yaxis=dict(title="Percentage of High Scores", range=[0, 110], ticksuffix='%'),
        xaxis=dict(title="Clinical Indicator"),
        height=350,
        margin=dict(l=60, r=40, t=60, b=80),
        showlegend=False
    )
    
    return fig


def create_score_gauge(score_percentage: float, title: str) -> go.Figure:
    """Create a gauge chart for overall score."""
    # Determine color based on score
    if score_percentage >= 75:
        color = "#48bb78"
        status = "Typical"
    elif score_percentage >= 50:
        color = "#ecc94b"
        status = "Monitor"
    elif score_percentage >= 25:
        color = "#ed8936"
        status = "Concern"
    else:
        color = "#f56565"
        status = "Evaluate"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14}},
        number={'suffix': '%', 'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': '#fed7d7'},
                {'range': [25, 50], 'color': '#feebc8'},
                {'range': [50, 75], 'color': '#fefcbf'},
                {'range': [75, 100], 'color': '#c6f6d5'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': score_percentage
            }
        }
    ))
    
    fig.add_annotation(
        x=0.5, y=-0.15,
        text=f"Status: {status}",
        showarrow=False,
        font=dict(size=12, color=color, weight='bold')
    )
    
    fig.update_layout(
        height=250,
        margin=dict(l=30, r=30, t=50, b=50)
    )
    
    return fig


def render_visual_report(child_name: str, child_age: str, all_responses: Dict[str, List[Dict]], ai_analysis: str):
    """Render the comprehensive visual assessment report."""
    
    # Calculate all scores and analysis
    domain_scores = calculate_domain_scores(all_responses)
    tag_analysis = analyze_clinical_tags(all_responses)
    
    # Calculate overall score
    if domain_scores:
        total_score = sum(d['total_score'] for d in domain_scores.values())
        total_max = sum(d['max_possible'] for d in domain_scores.values())
        overall_percentage = (total_score / total_max * 100) if total_max > 0 else 0
    else:
        overall_percentage = 0
    
    # Professional report header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%); 
                padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem; text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h1 style="color: white; margin: 0; font-size: 1.8rem;">📋 Developmental Screening Report</h1>
        <p style="color: #90cdf4; margin: 0.5rem 0 0 0; font-size: 1rem;">Comprehensive Multi-Modal Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Patient information cards
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.markdown(f"""
        <div style="background: #f0fff4; padding: 1rem; border-radius: 8px; text-align: center; 
                    border-left: 4px solid #38a169;">
            <p style="margin: 0; color: #666; font-size: 0.75rem; text-transform: uppercase;">Child</p>
            <p style="margin: 0; font-weight: bold; color: #1a202c; font-size: 1.1rem;">{child_name}</p>
        </div>
        """, unsafe_allow_html=True)
    with info_col2:
        st.markdown(f"""
        <div style="background: #ebf8ff; padding: 1rem; border-radius: 8px; text-align: center;
                    border-left: 4px solid #3182ce;">
            <p style="margin: 0; color: #666; font-size: 0.75rem; text-transform: uppercase;">Age</p>
            <p style="margin: 0; font-weight: bold; color: #1a202c; font-size: 1.1rem;">{child_age}</p>
        </div>
        """, unsafe_allow_html=True)
    with info_col3:
        st.markdown(f"""
        <div style="background: #faf5ff; padding: 1rem; border-radius: 8px; text-align: center;
                    border-left: 4px solid #805ad5;">
            <p style="margin: 0; color: #666; font-size: 0.75rem; text-transform: uppercase;">Date</p>
            <p style="margin: 0; font-weight: bold; color: #1a202c; font-size: 1.1rem;">{pd.Timestamp.now().strftime('%b %d, %Y')}</p>
        </div>
        """, unsafe_allow_html=True)
    with info_col4:
        total_qs = sum(len(resps) for resps in all_responses.values())
        st.markdown(f"""
        <div style="background: #fffaf0; padding: 1rem; border-radius: 8px; text-align: center;
                    border-left: 4px solid #dd6b20;">
            <p style="margin: 0; color: #666; font-size: 0.75rem; text-transform: uppercase;">Items Assessed</p>
            <p style="margin: 0; font-weight: bold; color: #1a202c; font-size: 1.1rem;">{total_qs} Questions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ===== VISUAL DASHBOARD SECTION =====
    st.markdown("### 📊 Assessment Dashboard")
    
    # Row 1: Overall gauge and domain radar
    gauge_col, radar_col = st.columns([1, 2])
    
    with gauge_col:
        overall_gauge = create_score_gauge(overall_percentage, "Overall Score")
        st.plotly_chart(overall_gauge, width='stretch')
    
    with radar_col:
        if len(domain_scores) >= 3:
            radar_chart = create_domain_radar_chart(domain_scores)
            st.plotly_chart(radar_chart, width='stretch')
        else:
            bar_chart = create_domain_bar_chart(domain_scores)
            st.plotly_chart(bar_chart, width='stretch')
    
    # Row 2: Domain bar chart and question types pie
    bar_col, pie_col = st.columns([2, 1])
    
    with bar_col:
        bar_chart = create_domain_bar_chart(domain_scores)
        st.plotly_chart(bar_chart, width='stretch')
    
    with pie_col:
        pie_chart = create_question_type_pie_chart(all_responses)
        st.plotly_chart(pie_chart, width='stretch')
    
    # Row 3: Learning Preferences breakdown (Auditory, Visual, Kinesthetic)
    st.markdown("#### 📚 Learning Style Breakdown")
    learning_prefs_chart = create_learning_preferences_chart(all_responses)
    st.plotly_chart(learning_prefs_chart, use_container_width=True)
    
    # Show learning style summary
    modality_data = analyze_learning_preferences(all_responses)
    if modality_data:
        dominant_style = max(modality_data.items(), key=lambda x: x[1]['percentage'])
        st.info(f"🎯 **Dominant Learning Style:** {dominant_style[0]} ({dominant_style[1]['percentage']:.0f}%) - Consider incorporating {dominant_style[0].lower()} activities in learning sessions.")
    
    # Row 4: Clinical flags
    if tag_analysis:
        flags_chart = create_clinical_flags_chart(tag_analysis)
        st.plotly_chart(flags_chart, width='stretch')
    
    st.markdown("---")
    
    # ===== DETAILED DOMAIN BREAKDOWN =====
    st.markdown("### 📈 Domain-by-Domain Analysis")
    
    for domain, scores in domain_scores.items():
        with st.expander(f"**{domain}** - Score: {scores['percentage']:.0f}%", expanded=False):
            # Mini gauge for this domain
            col1, col2 = st.columns([1, 2])
            
            with col1:
                mini_gauge = create_score_gauge(scores['percentage'], domain)
                st.plotly_chart(mini_gauge, width='stretch')
            
            with col2:
                st.markdown(f"""
                **Summary Statistics:**
                - Total Score: **{scores['total_score']}/{scores['max_possible']}**
                - Average per Question: **{scores['average']:.2f}**
                - Questions Assessed: **{scores['question_count']}**
                - High Scores (≥3): **{scores['high_scores']}**
                - Low Scores (≤1): **{scores['low_scores']}**
                """)
            
            # Show question details in a table
            if scores['details']:
                details_df = pd.DataFrame([
                    {
                        'Question': d['question'] + '...' if len(d['question']) >= 50 else d['question'],
                        'Type': d['type'].replace('_', ' ').title(),
                        'Response': str(d['response'])[:30],
                        'Score': f"{d['score']}/{d['max_score']}"
                    }
                    for d in scores['details']
                ])
                st.dataframe(details_df, width='stretch', hide_index=True)
    
    st.markdown("---")
    
    # ===== CLINICAL FLAGS SECTION =====
    st.markdown("### 🚩 Clinical Indicator Analysis")
    
    significant_flags = {
        tag: data for tag, data in tag_analysis.items()
        if data['total'] >= 2 and data['high_scores'] >= 1
    }
    
    if significant_flags:
        for tag, data in significant_flags.items():
            severity = "🔴 Significant" if data['high_scores'] >= 3 else "🟡 Moderate" if data['high_scores'] >= 2 else "🟢 Mild"
            
            # Improved visibility with darker text colors
            if data['high_scores'] >= 3:
                bg_color = '#fed7d7'
                border_color = '#c53030'
                text_color = '#742a2a'
                label_color = '#9b2c2c'
            elif data['high_scores'] >= 2:
                bg_color = '#fefcbf'
                border_color = '#d69e2e'
                text_color = '#744210'
                label_color = '#975a16'
            else:
                bg_color = '#c6f6d5'
                border_color = '#38a169'
                text_color = '#22543d'
                label_color = '#276749'
            
            st.markdown(f"""
            <div style="background: {bg_color}; 
                        padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;
                        border-left: 4px solid {border_color};">
                <div style="color: {text_color}; font-weight: bold; font-size: 1.1rem; margin-bottom: 0.3rem;">{tag}</div>
                <div style="color: {label_color}; font-weight: 600; margin-bottom: 0.3rem;">{severity}</div>
                <div style="color: {text_color}; font-size: 0.9rem;">{data['high_scores']} of {data['total']} items scored HIGH | Sections: {', '.join(data['sections'])}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No significant clinical flags identified in this screening.")
    
    st.markdown("---")
    
    # ===== AI CLINICAL INTERPRETATION =====
    st.markdown("### 🧠 Clinical Interpretation & Recommendations")
    st.markdown(ai_analysis)
    
    # Important disclaimer box
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%); 
                padding: 1.5rem; border-radius: 10px; border-left: 5px solid #c53030;
                margin: 1rem 0;">
        <h4 style="color: #c53030; margin-top: 0; margin-bottom: 0.5rem;">⚠️ IMPORTANT CLINICAL DISCLAIMER</h4>
        <p style="color: #742a2a; margin-bottom: 0.5rem; font-size: 0.95rem;">
            This developmental screening tool provides <strong>preliminary observations only</strong> and does <strong>NOT constitute a clinical diagnosis</strong>.
        </p>
        <ul style="color: #742a2a; margin-bottom: 0; padding-left: 1.5rem; font-size: 0.9rem;">
            <li>Results must be interpreted by qualified healthcare professionals</li>
            <li>Elevated scores suggest areas for further evaluation, not confirmed conditions</li>
            <li>This screening does not replace comprehensive clinical assessment</li>
            <li>Always consult your pediatrician or developmental specialist with any concerns</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    return domain_scores, tag_analysis, overall_percentage


def generate_assessment_prompt(child_name: str, child_age: str, responses: Dict) -> str:
    """Generate a clinically realistic assessment prompt for Gemini API."""
    
    # Calculate domain scores and analyze clinical tags
    domain_scores = {}
    tag_analysis = {}
    
    for section, section_responses in responses.items():
        scores = [r.get('score', 0) for r in section_responses if r.get('score') is not None]
        if scores:
            domain_scores[section] = {
                'total': sum(scores),
                'avg': sum(scores) / len(scores) if scores else 0,
                'count': len(scores),
                'high_count': sum(1 for s in scores if s >= 3)
            }
        
        # Analyze tags for clinical indicators (ADHD, Autism, IDD, etc.)
        for resp in section_responses:
            for tag in resp.get('tags', []):
                tag_clean = str(tag).strip().upper()
                if tag_clean:
                    if tag_clean not in tag_analysis:
                        tag_analysis[tag_clean] = {'high_scores': 0, 'total': 0, 'questions': []}
                    tag_analysis[tag_clean]['total'] += 1
                    if resp.get('score', 0) >= 3:  # HIGH score (Often/Always/Very much)
                        tag_analysis[tag_clean]['high_scores'] += 1
                        tag_analysis[tag_clean]['questions'].append(resp['question'][:50])
    
    prompt = f"""You are a licensed pediatric developmental specialist conducting a formal developmental screening assessment.

══════════════════════════════════════════════════════════════════════════════
                        DEVELOPMENTAL SCREENING DATA
══════════════════════════════════════════════════════════════════════════════

**PATIENT INFORMATION:**
• Child's Name: {child_name}
• Age at Assessment: {child_age}
• Assessment Date: {pd.Timestamp.now().strftime('%B %d, %Y')}
• Screening Type: Comprehensive Multi-Modal Developmental Screening

**QUANTITATIVE DOMAIN SCORES:**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    for domain, data in domain_scores.items():
        avg_pct = (data['avg'] / 4) * 100 if data['avg'] else 0
        flag = "⚠️ ELEVATED" if data['high_count'] >= 3 else "✓ TYPICAL"
        prompt += f"• {domain}: Average {data['avg']:.2f}/4.0 ({avg_pct:.0f}%) | HIGH scores: {data['high_count']}/{data['count']} {flag}\n"
    
    prompt += """

**CLINICAL TAG ANALYSIS (CRITICAL FOR DIAGNOSIS INDICATORS):**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    clinical_flags = []
    for tag, data in tag_analysis.items():
        if data['total'] > 0:
            if data['high_scores'] >= 2:
                severity = "🔴 SIGNIFICANT CONCERN" if data['high_scores'] >= 3 else "🟡 MODERATE CONCERN"
                prompt += f"• {tag}: {data['high_scores']}/{data['total']} items scored HIGH {severity}\n"
                clinical_flags.append(f"{tag} ({data['high_scores']}/{data['total']} HIGH)")
            elif data['high_scores'] == 1:
                prompt += f"• {tag}: {data['high_scores']}/{data['total']} items scored HIGH (Monitor)\n"
    
    if not clinical_flags:
        prompt += "• No significant clinical flags identified at this time\n"
    else:
        prompt += f"\n⚠️ **FLAGS REQUIRING ATTENTION:** {', '.join(clinical_flags)}\n"
    
    prompt += """

**DETAILED ITEM-BY-ITEM ANALYSIS:**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    for section, section_responses in responses.items():
        prompt += f"\n## {section.upper()} DOMAIN\n"
        prompt += "─" * 50 + "\n"
        
        for i, resp in enumerate(section_responses, 1):
            score = resp.get('score', 0)
            q_type = resp.get('question_type', 'observation')
            tags = resp.get('tags', [])
            
            # Score severity flag
            if score >= 3:
                score_flag = "⚠️ HIGH"
            elif score <= 1:
                score_flag = "✓ LOW"
            else:
                score_flag = "→ MODERATE"
            
            prompt += f"""
**Item {i}** [{q_type.upper()}] {score_flag}
Question: {resp['question']}
Response: {resp.get('response', 'N/A')} | Score: {score}/4
Clinical Tags: {', '.join(tags) if tags else 'General developmental'}
"""
            
            # Audio task specific analysis
            if q_type == 'audio_instruction':
                if resp.get('audio_played'):
                    prompt += f"   📢 Auditory Stimulus Given: \"{resp['audio_played']}\"\n"
                if resp.get('transcription'):
                    prompt += f"   🗣️ Child's Verbal Response: {resp['transcription']}\n"
                    prompt += f"   📊 Compare stimulus vs response for auditory processing assessment\n"
                if resp.get('has_recording'):
                    prompt += f"   🎙️ Audio Recording: Available\n"
            
            # Visual task specific analysis
            elif q_type == 'visual_identification':
                if resp.get('details'):
                    prompt += f"   👁️ Child's Visual Response: {resp['details']}\n"
            
            # Motor task specific analysis
            elif q_type == 'motor_task':
                if resp.get('attempts'):
                    prompt += f"   🔄 Trials Required: {resp['attempts']}\n"
                if resp.get('consistency'):
                    prompt += f"   📈 Performance Consistency: {resp['consistency']}\n"
                if resp.get('notes'):
                    prompt += f"   📝 Clinical Observation: {resp['notes']}\n"
            
            # Short answer analysis
            elif q_type == 'short_answer':
                if resp.get('child_answer'):
                    prompt += f"   💬 Child's Answer: {resp['child_answer']}\n"
                if resp.get('quality'):
                    prompt += f"   ✅ Response Quality: {resp['quality']}\n"
            
            # General details/notes
            if resp.get('details') and q_type not in ['visual_identification', 'short_answer']:
                prompt += f"   📋 Additional Details: {resp['details']}\n"
            if resp.get('notes') and q_type != 'motor_task':
                prompt += f"   📝 Notes: {resp['notes']}\n"
    
    prompt += """

══════════════════════════════════════════════════════════════════════════════
                    CLINICAL REPORT GENERATION INSTRUCTIONS
══════════════════════════════════════════════════════════════════════════════

Based on the assessment data above, generate a REALISTIC and CLINICALLY ACCURATE developmental screening report:

---

## 📊 EXECUTIVE SUMMARY
Write 3-4 sentences providing:
- Overall developmental status relative to age expectations
- Key strengths identified during assessment
- Primary areas of concern (if any)
- Overall clinical impression

## 💪 DEVELOPMENTAL STRENGTHS
List SPECIFIC areas where the child demonstrates:
- Age-appropriate or advanced skills
- Positive behavioral patterns observed
- Notable capabilities in specific domains
- Areas where scores were consistently LOW (indicating typical development for frequency-based concerns)

## 🎯 AREAS REQUIRING SUPPORT
Identify SPECIFIC concerns based on HIGH scores:
- Skills that appear below age expectations
- Behavioral patterns requiring monitoring
- Specify severity: Mild Concern / Moderate Concern / Significant Concern
- Reference specific items and scores that indicate these concerns

## 🧠 DOMAIN-BY-DOMAIN CLINICAL ANALYSIS

### Behavioral & Emotional Regulation
Analyze based on Behavioral domain scores:
- Attention span and sustained focus
- Impulse control and self-regulation  
- Emotional responses and regulation
- Social interaction patterns
- Hyperactivity indicators (if present)

### Cognitive & Communication
Analyze based on Cognitive domain and audio tasks:
- Language comprehension abilities
- Expressive language (analyze audio transcriptions)
- Memory and recall
- Auditory processing: COMPARE what was played vs what child said
- Problem-solving indicators

### Learning Profile
Analyze based on Learning Preferences domain:
- Preferred learning modalities (visual, auditory, kinesthetic)
- Engagement patterns and attention to tasks
- Environmental preferences

### Motor & Sensory Processing
Analyze based on Sensory & Physical domain:
- Gross motor coordination and skills
- Fine motor abilities
- Sensory processing patterns
- Physical task completion rates

## ⚠️ CLINICAL FLAGS & DIFFERENTIAL CONSIDERATIONS

**THIS SECTION IS CRITICAL - ANALYZE THE TAG DATA CAREFULLY:**

For each clinical tag with HIGH scores, provide specific analysis:

**If ADHD-tagged items show 2+ HIGH scores:**
- State: "X of Y ADHD-indicator items scored HIGH (Often/Always)"
- Discuss specific attention/hyperactivity/impulsivity patterns observed
- Recommend: "Formal ADHD evaluation by pediatric neurologist or developmental pediatrician is warranted"

**If AUTISM-tagged items show 2+ HIGH scores:**
- State: "X of Y Autism-indicator items scored HIGH"
- Discuss specific social communication or repetitive behavior patterns
- Recommend: "Autism spectrum screening by qualified specialist is recommended"

**If IDD (Intellectual/Developmental Disability) tagged items show patterns:**
- State the specific concerns observed
- Recommend appropriate developmental evaluation

**If SENSORY-tagged items are elevated:**
- Discuss sensory processing observations
- Recommend occupational therapy evaluation if warranted

**If no significant clinical flags:** State that no immediate clinical concerns were identified but continue routine developmental monitoring.

## 📋 RECOMMENDATIONS

### Immediate Home-Based Interventions
Provide 3-5 SPECIFIC, actionable activities parents can implement immediately based on the areas needing support.

### Professional Referrals (based on clinical flags identified)
Only recommend if indicated by the data:
- **Developmental Pediatrician**: If overall developmental delays noted
- **Pediatric Neurologist**: If ADHD indicators elevated
- **Child Psychologist**: If behavioral/emotional concerns
- **Speech-Language Pathologist**: If communication concerns (based on audio task performance)
- **Occupational Therapist**: If motor or sensory concerns
- **Audiologist**: If auditory processing concerns (based on poor audio task performance)

### Monitoring & Follow-up Timeline
- Specific behaviors to monitor over next 3-6 months
- When to seek reassessment
- Red flags that should prompt immediate evaluation

## 👨‍👩‍👧 PARENT SUMMARY

Write a supportive, parent-friendly summary that:
- Explains findings in simple, non-clinical language
- Provides appropriate reassurance where warranted
- Does NOT minimize genuine concerns
- Emphasizes that screening ≠ diagnosis
- Encourages follow-up without causing unnecessary anxiety
- Highlights the child's strengths

---

## ⚠️ MANDATORY DISCLAIMER
Include this statement: "This developmental screening provides preliminary observations only and does NOT constitute a clinical diagnosis. All findings should be reviewed by qualified healthcare professionals. Elevated scores on clinical indicators suggest areas warranting further professional evaluation, not confirmed diagnoses."

---

**CRITICAL INSTRUCTIONS FOR GENERATING THIS REPORT:**
1. Be REALISTIC - If scores genuinely indicate concerns, clearly state them
2. Be SPECIFIC - Reference actual scores, specific items, and observed patterns
3. Be BALANCED - Acknowledge both strengths and concerns honestly
4. ANALYZE CLINICAL TAGS - This is crucial for identifying ADHD/Autism/IDD indicators
5. For AUDIO tasks - Specifically compare the stimulus given vs child's response
6. QUANTIFY concerns - "3 of 4 ADHD items scored HIGH" not "some ADHD concerns"
7. Make ACTIONABLE recommendations based on findings
8. Use professional clinical language while remaining parent-accessible
"""
    
    return prompt


def call_gemini_api(api_key: str, prompt: str) -> str:
    """Call Gemini API."""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"


def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.markdown("## 🏥 Settings")
        st.markdown("---")
        
        # Get API key from Streamlit secrets
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        
        if api_key:
            st.success("✅ API Key loaded from secrets")
        else:
            st.warning("⚠️ Add GEMINI_API_KEY to Streamlit secrets")
        
        st.markdown("---")
        st.markdown("### 👶 Child Information")
        
        child_name = st.text_input("Child's Name", placeholder="Name")
        child_age = st.text_input("Child's Age", placeholder="e.g., 4 years")
        
        st.markdown("---")
        st.markdown("### 📋 Question Types")
        st.markdown("""
        - 📊 Frequency Ratings
        - ✏️ Short Answers
        - 🎧 Audio Tasks
        - 🖼️ Visual Tasks
        - 🏃 Motor Skills
        - ✅ Yes/No
        - 💡 Preferences
        """)
        
        return api_key, child_name, child_age


def main():
    """Main application."""
    initialize_session_state()
    
    api_key, child_name, child_age = render_sidebar()
    
    st.title("🧒 Child Development Screening Tool")
    st.caption("Multi-modal developmental assessment with AI-powered analysis")
    
    if not api_key or not child_age:
        st.info("👈 Enter **API Key** and **Child's Age** to begin.")
        
        st.markdown("### 📚 Assessment Types")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            🎧 **Audio Tasks** - Auditory processing  
            🖼️ **Visual Tasks** - Visual perception  
            🏃 **Motor Tasks** - Coordination  
            """)
        
        with col2:
            st.markdown("""
            📊 **Frequency Ratings** - Behaviors  
            ✏️ **Short Answers** - Comprehension  
            💡 **Preferences** - Learning style  
            """)
        return
    
    # Load questions
    if not st.session_state.all_questions_raw:
        with st.spinner("Loading..."):
            st.session_state.all_questions_raw = load_all_questions_raw()
    
    if not st.session_state.all_questions_raw:
        st.error("❌ Cannot load questions. Check data files.")
        return
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        load_button = st.button("🔄 Load New Random Questions", width='stretch', 
                                help="Click to get a fresh random set of age-appropriate questions")
    
    if load_button or not st.session_state.questions_loaded or st.session_state.child_age_for_questions != child_age:
        # Generate new random seed for each load to ensure different questions
        st.session_state.random_seed = int(time.time() * 1000) % (2**32)
        
        # Reload raw questions with new random order
        with st.spinner("Loading fresh questions..."):
            st.session_state.all_questions_raw = load_all_questions_raw()
        
        with st.spinner(f"🤖 Randomly selecting questions for {child_age} old..."):
            st.session_state.all_questions = select_age_appropriate_questions_with_ai(
                api_key, child_age, st.session_state.all_questions_raw, num_questions=10
            )
            st.session_state.questions_loaded = True
            st.session_state.child_age_for_questions = child_age
            st.session_state.assessment_generated = False
            st.session_state.audio_cache = {}
            st.session_state.image_cache = {}
            if load_button:
                st.rerun()
    
    if not st.session_state.all_questions:
        st.warning("No questions selected. Try again.")
        return
    
    # Determine age group for display
    age_lower = child_age.lower()
    if any(x in age_lower for x in ['month', '0 year', '1 year']) and 'year' not in age_lower.replace('1 year', ''):
        age_group = "Infant/Toddler"
    elif any(x in age_lower for x in ['2 year', '3 year']):
        age_group = "Toddler"
    elif any(x in age_lower for x in ['3 year', '4 year', '5 year']):
        age_group = "Preschooler"
    elif any(x in age_lower for x in ['5 year', '6 year', '7 year']):
        age_group = "Early School Age"
    else:
        age_group = "School Age"
    
    st.success(f"✅ Loaded **randomized age-appropriate questions** for **{child_age}** ({age_group} developmental stage)")
    st.caption(f"🎲 Session ID: {st.session_state.get('random_seed', 0)} - Click 'Load New Random Questions' for different questions")
    
    tabs = st.tabs(list(DATA_FILES.keys()))
    all_responses = {}
    
    for tab, (section_name, _) in zip(tabs, DATA_FILES.items()):
        with tab:
            questions = st.session_state.all_questions.get(section_name, [])
            section_key = section_name.replace(" ", "_").replace("&", "and")
            responses = render_question_section(section_name, questions, section_key, api_key)
            all_responses[section_name] = responses
    
    st.markdown("---")
    st.markdown("### 📊 Generate Assessment")
    
    # Check if all questions are answered
    all_answered, answered_count, total_count, missing_sections = check_all_questions_answered(all_responses)
    
    validation_errors = []
    if not child_name:
        validation_errors.append("Child's name required")
    
    if not all_answered:
        validation_errors.append(f"Please answer all questions ({answered_count}/{total_count} completed)")
    
    # Show progress
    if total_count > 0:
        progress = answered_count / total_count
        st.progress(progress, text=f"Assessment Progress: {answered_count}/{total_count} questions answered ({progress*100:.0f}%)")
        
        if missing_sections:
            st.info(f"📝 Sections with unanswered questions: {', '.join(missing_sections)}")
    
    if validation_errors:
        for error in validation_errors:
            st.warning(f"⚠️ {error}")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_button = st.button(
            "🎯 Generate Assessment Report", 
            width='stretch', 
            disabled=bool(validation_errors),
            help="Complete all questions in every section to enable this button"
        )
    
    if generate_button and not validation_errors:
        # Store responses in session state for the report
        st.session_state.all_responses_for_report = all_responses
        
        with st.spinner("🔄 Analyzing responses and generating comprehensive report..."):
            prompt = generate_assessment_prompt(child_name, child_age, all_responses)
            assessment = call_gemini_api(api_key, prompt)
            st.session_state.assessment_result = assessment
            st.session_state.assessment_generated = True
            st.rerun()
    
    if st.session_state.assessment_generated and st.session_state.assessment_result:
        st.markdown("---")
        
        # Use stored responses for report generation
        report_responses = st.session_state.get('all_responses_for_report', all_responses)
        
        # Render the visual report with charts and graphs
        domain_scores, tag_analysis, overall_percentage = render_visual_report(
            child_name, 
            child_age, 
            report_responses,
            st.session_state.assessment_result
        )
        
        # Export options
        st.markdown("### 📥 Export Report")
        
        # Prepare detailed report content
        report_content = f"""
══════════════════════════════════════════════════════════════════════════════
                    DEVELOPMENTAL SCREENING REPORT
══════════════════════════════════════════════════════════════════════════════

PATIENT INFORMATION
━━━━━━━━━━━━━━━━━━━
Child's Name: {child_name}
Age at Assessment: {child_age}
Assessment Date: {pd.Timestamp.now().strftime('%B %d, %Y')}
Screening Type: Comprehensive Multi-Modal Developmental Screening
Overall Score: {overall_percentage:.1f}%

══════════════════════════════════════════════════════════════════════════════
                        DOMAIN SCORES SUMMARY
══════════════════════════════════════════════════════════════════════════════
"""
        for domain, scores in domain_scores.items():
            report_content += f"""
{domain}:
  - Score: {scores['total_score']}/{scores['max_possible']} ({scores['percentage']:.1f}%)
  - Average: {scores['average']:.2f}
  - High Scores: {scores['high_scores']}/{scores['question_count']}
"""

        report_content += f"""
══════════════════════════════════════════════════════════════════════════════
                        CLINICAL INTERPRETATION
══════════════════════════════════════════════════════════════════════════════

{st.session_state.assessment_result}

══════════════════════════════════════════════════════════════════════════════
                            DISCLAIMER
══════════════════════════════════════════════════════════════════════════════

This developmental screening tool provides preliminary observations only and 
does NOT constitute a clinical diagnosis. All findings should be reviewed by 
qualified healthcare professionals. Elevated scores on clinical indicators 
suggest areas warranting further professional evaluation, not confirmed 
diagnoses.

Always consult your pediatrician or developmental specialist for clinical 
evaluation and guidance.

══════════════════════════════════════════════════════════════════════════════
Report Generated: {pd.Timestamp.now().strftime('%B %d, %Y at %I:%M %p')}
Tool: Child Development Screening Tool v1.0
══════════════════════════════════════════════════════════════════════════════
"""
        
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                "📄 Download Full Report (TXT)",
                data=report_content,
                file_name=f"developmental_screening_{child_name.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                width='stretch'
            )
        with export_col2:
            st.info("💡 **Tip:** Press Ctrl+P (Cmd+P on Mac) to print this page as a formatted PDF")


if __name__ == "__main__":
    main()
