"""Actual pinned Dory header plus malformed-metadata negative controls."""
import copy
import sys
import unittest
from pathlib import Path
import inspect_dory_checkpoint_metadata_v1 as d

class Checkpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=d.Q/'dory-checkpoint-metadata-20261108-v1'
        cls.raw=(p/'safetensors-header.json').read_bytes()
        cls.header=d.parse(cls.raw)
        cls.config=d.DoryContract.from_mapping(d.parse((p/'config.json').read_bytes()))
        cls.expected=cls.config.weight_shapes()
        cls.size=7950515472
    def check(self,change=None,size=None):
        h=copy.deepcopy(self.header)
        if change:change(h)
        return d.checkpoint(h,self.expected,len(self.raw),self.size if size is None else size)
    def test_actual_507_physical_weights(self):
        r=self.check()
        self.assertEqual((r['tensors'],r['parameters'],r['data_bytes']),(507,3975229440,7950458880))
        self.assertFalse(r['weight_values_read'])
    def test_missing_learned_logit_scale_rejected(self):
        with self.assertRaises(ValueError):self.check(lambda h:h.pop('lm_head.logit_scale'))
    def test_virtual_loop_weights_rejected(self):
        with self.assertRaises(ValueError):self.check(lambda h:h.update({'backbone.layers.72.mixer.q_proj.weight':h['backbone.layers.0.mixer.q_proj.weight']}))
    def test_wrong_attention_width_rejected(self):
        with self.assertRaises(ValueError):self.check(lambda h:h['backbone.layers.0.mixer.q_proj.weight'].update(shape=[2560,2560]))
    def test_quantized_dtype_rejected(self):
        with self.assertRaises(ValueError):self.check(lambda h:h['lm_head.weight'].update(dtype='I8'))
    def test_bool_dimension_rejected(self):
        with self.assertRaises(ValueError):self.check(lambda h:h['lm_head.logit_scale'].update(shape=[True]))
    def test_missing_tensor_bytes_rejected(self):
        def change(h):h['lm_head.logit_scale']['data_offsets'][1]-=2
        with self.assertRaises(ValueError):self.check(change)
    def test_overlap_rejected(self):
        def change(h):
            name=next(k for k,v in h.items()if v['data_offsets'][0]>0)
            h[name]['data_offsets']=[x-2 for x in h[name]['data_offsets']]
        with self.assertRaises(ValueError):self.check(change)
    def test_gap_rejected(self):
        def change(h):
            name=next(k for k,v in h.items()if v['data_offsets'][0]>0)
            h[name]['data_offsets']=[x+2 for x in h[name]['data_offsets']]
        with self.assertRaises(ValueError):self.check(change)
    def test_trailing_payload_rejected(self):
        with self.assertRaises(ValueError):self.check(size=self.size+2)
    def test_short_payload_rejected(self):
        with self.assertRaises(ValueError):self.check(size=self.size-2)
    def test_duplicate_json_name_rejected(self):
        with self.assertRaises(ValueError):d.parse('{"x": 1, "x": 2}')
    def test_unexpected_tensor_attribute_rejected(self):
        with self.assertRaises(ValueError):self.check(lambda h:h['lm_head.weight'].update(scale=0.1))
    def test_original_recurrence_and_distinct_histories(self):
        c=self.config
        self.assertEqual((len(c.visits()),len(c.cache_plan())),(112,56))
        self.assertEqual(len({x['cache_id']for x in c.cache_plan()}),56)
        self.assertEqual(c.n_recurrent_loops,2)

class Tokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=d.Q/'dory-checkpoint-metadata-20261108-v1'
        cls.tok=d.parse((p/'tokenizer.json').read_bytes())
        cls.cfg=d.parse((p/'tokenizer_config.json').read_bytes())
        cls.special=d.parse((p/'special_tokens_map.json').read_bytes())
        cls.generation=d.parse((p/'generation_config.json').read_bytes())
    def check(self,change=None):
        values=copy.deepcopy((self.tok,self.cfg,self.special,self.generation))
        if change:change(*values)
        return d.tokenizer(*values,131072)
    def test_actual_complete_tokenizer_structure(self):
        r=self.check()
        self.assertEqual((r['vocabulary'],r['added_tokens'],r['merges']),(131072,1000,269443))
        self.assertEqual(r['stop_tokens'],[dict(id=2,token='</s>'),dict(id=11,token='<|im_end|>')])
        self.assertFalse(r['tokenizer_executed'])
    def test_duplicate_vocab_id_rejected(self):
        with self.assertRaises(ValueError):self.check(lambda t,c,s,g:t['model']['vocab'].update({'<unk>':1}))
    def test_added_token_metadata_mismatch_rejected(self):
        with self.assertRaises(ValueError):self.check(lambda t,c,s,g:c['added_tokens_decoder']['12'].update(special=True))
    def test_added_token_id_mismatch_rejected(self):
        with self.assertRaises(ValueError):self.check(lambda t,c,s,g:t['added_tokens'][12].update(id=999999))
    def test_missing_eos_rejected(self):
        with self.assertRaises(ValueError):self.check(lambda t,c,s,g:g.update(eos_token_id=[2]))
    def test_out_of_range_stop_rejected(self):
        with self.assertRaises(ValueError):self.check(lambda t,c,s,g:g.update(eos_token_id=[2,11,131072]))
    def test_wrong_padding_id_rejected(self):
        with self.assertRaises(ValueError):self.check(lambda t,c,s,g:g.update(pad_token_id=2))
    def test_missing_merge_symbol_rejected(self):
        with self.assertRaises(ValueError):self.check(lambda t,c,s,g:t['model']['merges'].append(['not-a-real-token-xyz','not-a-real-token-abc']))
    def test_no_framework_loaded(self):
        self.assertFalse({'torch','tokenizers','transformers','numpy'} & set(sys.modules))

if __name__=='__main__':unittest.main(verbosity=2)
