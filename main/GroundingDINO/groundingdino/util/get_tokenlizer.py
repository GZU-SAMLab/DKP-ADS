from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast


# def get_tokenlizer(text_encoder_type):
#     if not isinstance(text_encoder_type, str):
#         # print("text_encoder_type is not a str")
#         if hasattr(text_encoder_type, "text_encoder_type"):
#             text_encoder_type = text_encoder_type.text_encoder_type
#         elif text_encoder_type.get("text_encoder_type", False):
#             text_encoder_type = text_encoder_type.get("text_encoder_type")
#         else:
#             raise ValueError(
#                 "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
#             )
#     print("final text_encoder_type: {}".format(text_encoder_type))
#
#     tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
#     return tokenizer

def get_tokenlizer(text_encoder_type):
    # 假设你已经下载了分词器的文件，并将它们放在了这个路径下
    tokenizer_path = "./bert-base-uncased/"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer


# def get_pretrained_language_model(text_encoder_type):
#     if text_encoder_type == "bert-base-uncased":
#         return BertModel.from_pretrained(text_encoder_type)
#     if text_encoder_type == "roberta-base":
#         return RobertaModel.from_pretrained(text_encoder_type)
#     raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))

def get_pretrained_language_model(text_encoder_type):
    # 假设你已经下载了模型的文件，并将它们放在了这个路径下
    model_path = "./bert-base-uncased/"
    if text_encoder_type == "bert-base-uncased":
        return BertModel.from_pretrained(model_path)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(model_path)
    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
