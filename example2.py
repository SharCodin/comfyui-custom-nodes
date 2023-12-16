class ConditioningPrompts:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "pos_text": ("STRING", {
                    "multiline": True,
                    "default": "Positive text"
                }),
                "neg_text": ("STRING", {
                    "multiline": True,
                    "default": "Negative text"
                })
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("positive", "negative")
    
    FUNCTION = "test"

    CATEGORY = "Example2"

    def test(self, clip, pos_text, neg_text):
        tokens_pos = clip.tokenize(pos_text)
        cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
        
        tokens_neg = clip.tokenize(neg_text)
        cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
        
        return (
            [[cond_pos, {"pooled_output": pooled_pos}]], 
            [[cond_neg, {"pooled_output": pooled_neg}]], 
        )




NODE_CLASS_MAPPINGS = {
    "Example2": ConditioningPrompts
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Example2": "Conditioning Prompts"
}
