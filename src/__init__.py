from .bnftocode import Quantizer

def build_model(model_name: str):
    if model_name == "vector-quantizer":
        return Quantizer
    else:
        raise ValueError(f"Unknown model name: {model_name}.")
