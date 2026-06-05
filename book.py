from llm_sdk import Small_LLM_Model as llm_model


if __name__ == "__main__":
    model = llm_model()
    tokenizer = model._tokenizer
    gwen_dict = model.get_path_to_vocab_file()
    sample_str = 'sum'
    result = model.encode(sample_str).tolist()[0]
    print(result)
    nbr = [30337]
    print(f"'{model.decode(nbr)}'")

# '",\n' | 756
# ' {"' | 5212
# '"}' | 9207
# '":"' | 3252
# '","' | 2198
# '":{"' | 22317
# ',"' | 1335
