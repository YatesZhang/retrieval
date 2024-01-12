def _convert_2_vocabs(text, tokenizer, vocabs):
    """ 
        skip tokens out of vocabs
    """
    assert isinstance(text, str)
    _input_ids = tokenizer(text)['input_ids']
    input_ids = []·
    for token_id in _input_ids:
        if token_id in vocabs:
            input_ids.append(token_id)
    return input_ids

def convert_in_vocabs(texts, tokenizer, vocabs):
    """ 
    """
    result = []
    if isinstance(texts, list):
        for text in texts:
            result.append(_convert_2_vocabs(text, tokenizer, vocabs))
    elif isinstance(texts, str):
        result.append(_convert_2_vocabs(texts, tokenizer, vocabs))
    return tokenizer.batch_decode(result)