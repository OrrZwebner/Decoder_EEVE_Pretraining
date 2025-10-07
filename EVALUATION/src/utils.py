"""
Utility functions for FLORES translation evaluation
"""

import os
import logging
from datetime import datetime
from pathlib import Path


# FLORES-200 Language Code Mapping
FLORES_LANG_MAP = {
    # Major European Languages
    "English": "eng_Latn",
    "Spanish": "spa_Latn", 
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "Dutch": "nld_Latn",
    "Russian": "rus_Cyrl",
    "Polish": "pol_Latn",
    "Ukrainian": "ukr_Cyrl",
    "Czech": "ces_Latn",
    "Greek": "ell_Grek",
    "Romanian": "ron_Latn",
    "Hungarian": "hun_Latn",
    "Swedish": "swe_Latn",
    "Danish": "dan_Latn",
    "Norwegian": "nob_Latn",
    "Finnish": "fin_Latn",
    
    # Asian Languages
    "Chinese (Simplified)": "zho_Hans",
    "Chinese (Traditional)": "zho_Hant", 
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Vietnamese": "vie_Latn",
    "Thai": "tha_Thai",
    "Indonesian": "ind_Latn",
    "Malay": "zsm_Latn",
    "Filipino": "fil_Latn",
    "Burmese": "mya_Mymr",
    "Khmer": "khm_Khmr",
    "Lao": "lao_Laoo",
    
    # Indian Subcontinent Languages
    "Hindi": "hin_Deva",
    "Sanskrit": "san_Deva",
    "Bengali": "ben_Beng",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Punjabi": "pan_Guru",
    "Urdu": "urd_Arab",
    "Nepali": "nep_Deva",
    "Sinhala": "sin_Sinh",
    "Odia": "ory_Orya",
    "Assamese": "asm_Beng",
    
    # Middle Eastern Languages
    "Arabic": "arb_Arab",
    "Hebrew": "heb_Hebr",
    "Turkish": "tur_Latn",
    "Persian": "pes_Arab",
    "Pashto": "pus_Arab",
    "Kurdish": "ckb_Arab",
    
    # African Languages
    "Swahili": "swh_Latn",
    "Yoruba": "yor_Latn",
    "Igbo": "ibo_Latn",
    "Zulu": "zul_Latn",
    "Xhosa": "xho_Latn",
    "Hausa": "hau_Latn",
    "Amharic": "amh_Ethi",
    "Somali": "som_Latn",
    "Afrikaans": "afr_Latn",
    
    # Other Languages
    "Estonian": "est_Latn",
    "Latvian": "lav_Latn",
    "Lithuanian": "lit_Latn",
    "Albanian": "sqi_Latn",
    "Armenian": "hye_Armn",
    "Georgian": "kat_Geor",
    "Basque": "eus_Latn",
    "Catalan": "cat_Latn",
    "Galician": "glg_Latn",
    "Welsh": "cym_Latn",
    "Irish": "gle_Latn",
    "Scottish Gaelic": "gla_Latn",
    "Icelandic": "isl_Latn",
    "Luxembourgish": "ltz_Latn",
    "Maltese": "mlt_Latn",
    "Macedonian": "mkd_Cyrl",
    "Bulgarian": "bul_Cyrl",
    "Serbian": "srp_Cyrl",
    "Croatian": "hrv_Latn",
    "Bosnian": "bos_Latn",
    "Slovenian": "slv_Latn",
    "Slovak": "slk_Latn",
    "Belarusian": "bel_Cyrl",
    "Kazakh": "kaz_Cyrl",
    "Uzbek": "uzb_Latn",
    "Azerbaijani": "azj_Latn",
    "Tajik": "tgk_Cyrl",
    "Mongolian": "mon_Cyrl",
}


def get_flores_lang_map():
    """Return FLORES language mapping dictionary"""
    return FLORES_LANG_MAP


def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime("%d%m%Y_%H%M")


def setup_logging(log_file=None):
    """
    Setup logging for nohup execution with immediate flushing
    Outputs to stdout (captured by nohup)
    
    Args:
        log_file: Not used, kept for compatibility
        
    Returns:
        Logger instance
    """
    import sys
    
    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    # Create logger
    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Console handler (captured by nohup)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler
    logger.addHandler(console_handler)
    
    return logger


def create_output_directory(config):
    """
    Create output directory for results
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to output directory
    """
    base_dir = config['paths']['output_dir']
    
    # Create experiment name
    provider_type = config['provider']['type']
    model_name = config['provider'][provider_type]['model_name'].replace('/', '_')
    source_lang = config['translation']['source_lang']
    target_lang = config['translation']['target_lang']
    timestamp = get_timestamp()
    
    experiment_name = f"{model_name}_{source_lang}-{target_lang}_{timestamp}"
    output_dir = os.path.join(base_dir, experiment_name)
    
    # Create directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    return output_dir


def get_flores_code(lang_name):
    """
    Get FLORES code from language name
    
    Args:
        lang_name: Language name or FLORES code
        
    Returns:
        FLORES code string
    """
    # If already a FLORES code (contains underscore), return as-is
    if "_" in lang_name:
        # verify it's in the mapping values
        if lang_name not in FLORES_LANG_MAP.values():
            raise ValueError(
                f"FLORES code '{lang_name}' not found in FLORES mapping values. "
                f"Please check the code or use a language name."
            )
            return None # unreachable
        return lang_name # valid FLORES code
    
    # Look up in mapping
    flores_code = FLORES_LANG_MAP.get(lang_name)
    
    if flores_code is None:
        raise ValueError(
            f"Language '{lang_name}' not found in FLORES mapping. "
            f"Use a FLORES code (e.g., 'xxx_Latn') or add to FLORES_LANG_MAP."
        )
    
    return flores_code