"""
Translation logic using LangChain with hardcoded prompt
"""

import torch
from tqdm import tqdm
from langchain.prompts import ChatPromptTemplate


# Hardcoded prompt template for translation
TRANSLATION_PROMPT_SYSTEM = """You are a professional translator. Your task is to translate text accurately from one language to another.

Instructions:
- Translate the given text from the source language to the target language
- Provide ONLY the translation, no explanations or notes
- Return your output as JSON with a "translation" key
- The translation must be accurate and natural in the target language

Output format:
{{"translation": "your translated text here"}}

Only respond with the JSON output, do not include any additional text."""


TRANSLATION_PROMPT_USER = """Source language: {source_lang}
Target language: {target_lang}

Text to translate: {text}

Return only JSON:"""


def create_prompt_template():
    """
    Create LangChain prompt template with hardcoded prompts
    
    Returns:
        ChatPromptTemplate instance
    """
    messages = [
        ("system", TRANSLATION_PROMPT_SYSTEM),
        ("human", TRANSLATION_PROMPT_USER)
    ]
    
    prompt_template = ChatPromptTemplate.from_messages(messages)
    
    return prompt_template


def create_translation_chain(llm, config, output_parser):
    """
    Create translation chain using LCEL
    
    Args:
        llm: LangChain LLM instance
        config: Configuration dictionary (not used for prompt, kept for compatibility)
        output_parser: TranslationOutputParser instance
        
    Returns:
        LangChain chain
    """
    prompt_template = create_prompt_template()
    
    # Create chain: prompt | llm | parser
    chain = prompt_template | llm | output_parser
    
    return chain


def translate_text(chain, text, source_lang, target_lang, logger=None):
    """
    Translate single text using the chain
    
    Args:
        chain: LangChain translation chain
        text: Source text to translate
        source_lang: Source language name
        target_lang: Target language name
        logger: Logger instance for logging raw output
        
    Returns:
        Translated text string
    """
    # Get raw model output before parsing
    raw_output = None
    try:
        # Invoke the chain (prompt | llm)
        # We need to get intermediate output before parser
        prompt_template = chain.first
        llm = chain.middle[0] if hasattr(chain, 'middle') else None
        
        # Invoke to get raw output
        full_chain_result = chain.invoke({
            "source_lang": source_lang,
            "target_lang": target_lang,
            "text": text
        })
        
        result = full_chain_result
        
    except Exception as e:
        if logger:
            logger.error(f"Translation error: {str(e)}")
        raise
    
    return result


def translate_batch(chain, data_pairs, config, logger=None):
    """
    Translate batch of texts with progress bar and memory management
    
    Args:
        chain: LangChain translation chain
        data_pairs: List of translation pairs
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        List of translated strings
    """
    translations = []
    
    source_lang = config['translation']['source_lang']
    target_lang = config['translation']['target_lang']
    cleanup_interval = config['batch'].get('memory_cleanup_interval', 10)
    
    if logger:
        logger.info(f"\nTranslating {len(data_pairs)} texts...")
    else:
        print(f"\nTranslating {len(data_pairs)} texts...")
    
    for idx, pair in enumerate(tqdm(data_pairs, desc="Translation progress")):
        try:
            # Log source text
            if logger and idx < 3:  # Log first 3 for verification
                logger.info(f"\n--- Sample {idx} ---")
                logger.info(f"Source: {pair['source']}")
            
            translation = translate_text(
                chain,
                pair['source'],
                source_lang,
                target_lang,
                logger=logger
            )
            
            # Log translation result
            if logger and idx < 3:
                logger.info(f"Translation: {translation}")
            
            translations.append(translation)
        except Exception as e:
            error_msg = f"\nError translating text {idx}: {str(e)}"
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
            # Append empty string on error
            translations.append("")
        
        # Periodic GPU memory cleanup
        if torch.cuda.is_available() and (idx + 1) % cleanup_interval == 0:
            torch.cuda.empty_cache()
    
    return translations