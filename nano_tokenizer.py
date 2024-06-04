import torch

class NanoTokenizer:
    def __init__(self, padding_side='left'):
        self.bos_token_id = 2               # 序列开始标记的 token ID
        self.eos_token_id = 3               # 序列结束标记的 token ID
        self.pad_token_id = 0               # 填充标记的 token ID
        self.padding_side = padding_side    # 填充方向

    def __len__(self):
        return 256

    def encode(self, text, return_tensors=None):
        utf8_bytes = text.encode('utf-8')
        tokens = [byte for byte in utf8_bytes]
        
        attention_mask = [1] * len(tokens)  # 注意力掩码，1 表示有效 token
        
        if return_tensors == 'pt':
            tokens = torch.tensor(tokens, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return {'input_ids': tokens, 'attention_mask': attention_mask}

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            token_list = tokens.tolist()
        else:
            token_list = tokens
        utf8_bytes = bytes(token_list)
        text = utf8_bytes.decode('utf-8')
        return text

    def __call__(self, text, return_tensors=None, padding=False):
        if isinstance(text, str):
            return self.encode(text, return_tensors=return_tensors)
        elif isinstance(text, list):
            return self.batch_encode(text, return_tensors=return_tensors, padding=padding)
        else:
            raise ValueError("Input should be a string or a list of strings")

    def batch_encode(self, texts, return_tensors=None, padding=False):
        batch_tokens = [self.encode(text) for text in texts]
        
        input_ids = [item['input_ids'] for item in batch_tokens]
        attention_masks = [item['attention_mask'] for item in batch_tokens]
        
        if padding:
            max_len = max(len(tokens) for tokens in input_ids)
            
            if self.padding_side == 'right':
                input_ids = [tokens + [self.pad_token_id] * (max_len - len(tokens)) for tokens in input_ids]
                attention_masks = [mask + [0] * (max_len - len(mask)) for mask in attention_masks]
            elif self.padding_side == 'left':
                input_ids = [[self.pad_token_id] * (max_len - len(tokens)) + tokens for tokens in input_ids]
                attention_masks = [[0] * (max_len - len(mask)) + mask for mask in attention_masks]
            else:
                raise ValueError("padding_side should be 'left' or 'right'")
        
        if return_tensors == 'pt':
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        
        return {'input_ids': input_ids, 'attention_mask': attention_masks}

    def batch_decode(self, batch_tokens):
        if isinstance(batch_tokens, torch.Tensor):
            batch_tokens = batch_tokens.tolist()
        
        texts = [self.decode(tokens) for tokens in batch_tokens]
        return texts
