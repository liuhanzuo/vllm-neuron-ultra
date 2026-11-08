"""Process-local canonical MistralCommon contracts for old vLLM's tokenizer.

The underlying tekken file, byte ranks, regex and chat encoder stay original.
Old vLLM exposes a static special-token subset, a different lossy vocabulary
dictionary and BOS+EOS for explicit add_special_tokens=True. Match the pinned
HF MistralCommonBackend's inference-mode contracts; never modify the SDK files.
"""
def install():
    from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer
    from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
    if getattr(MistralTokenizer,'_trn1_ministral3_canonical',False):return
    original_init=MistralTokenizer.__init__
    def initialize(self,tokenizer):
        original_init(self,tokenizer)
        if not self.is_tekken:raise ValueError('Experimental canonical tokenizer requires original Tekken')
        core=self.tokenizer;shift=core.num_special_tokens
        def token_id(piece):
            encoded=piece.encode('utf-8')
            if encoded in core._tekken_token2id_nospecial:return shift+core._tekken_token2id_nospecial[encoded]
            return core._special_tokens_reverse_vocab.get(piece,core.unk_id)
        self._vocab_dict={piece:token_id(piece) for piece in core.vocab()}
    def encode(self,text,truncation=None,max_length=None,add_special_tokens=None):
        ids=self.tokenizer.encode(text,bos=True if add_special_tokens is None else add_special_tokens,eos=False)
        return ids[:max_length] if truncation and max_length is not None else ids
    def decode(self,ids,skip_special_tokens=True):
        if isinstance(ids,int):ids=[ids]
        policy=SpecialTokenPolicy.IGNORE if skip_special_tokens else SpecialTokenPolicy.KEEP
        return self.tokenizer.decode(ids,policy)
    MistralTokenizer.__init__=initialize
    MistralTokenizer.encode=encode
    MistralTokenizer.decode=decode
    MistralTokenizer.all_special_ids=property(lambda self:sorted(self.tokenizer._special_token_ids))
    MistralTokenizer.all_special_tokens_extended=property(lambda self:[x['token_str'] for x in sorted(self.tokenizer._all_special_tokens,key=lambda x:x['rank'])])
    MistralTokenizer._trn1_ministral3_canonical=True
