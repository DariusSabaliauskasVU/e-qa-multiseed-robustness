import run_experiments as R

class MockEnc(dict):
    def __init__(self, seq_ids_list, **kw):
        super().__init__(**kw); self._seq=seq_ids_list
    def sequence_ids(self, i): return self._seq[i]

def mock_tokenizer(questions, contexts, **kw):
    # one example -> two windows over context "AAAA BBBB CCCC DDDD"
    seq=[[0,0,0,1,1,0],[0,0,0,1,1,0]]
    offs=[[(0,0),(0,0),(0,0),(0,4),(5,9),(0,0)],
          [(0,0),(0,0),(0,0),(10,14),(15,19),(0,0)]]
    return MockEnc(seq,
        input_ids=[[101,1,102,10,11,102],[101,1,102,12,13,102]],
        attention_mask=[[1]*6,[1]*6],
        offset_mapping=offs,
        overflow_to_sample_mapping=[0,0])

ex={"question":["q"],"context":["AAAA BBBB CCCC DDDD"],
    "answers":[{"text":["CCCC"],"answer_start":[10]}]}
feats=R.prepare_train_features(ex, mock_tokenizer)
assert len(feats)==2
# window 0: answer NOT inside -> CLS (0,0)
assert feats[0]["start"]==0 and feats[0]["end"]==0, feats[0]
# window 1: answer inside -> tokens pointing at "CCCC" (index 3)
assert feats[1]["start"]==3 and feats[1]["end"]==3, feats[1]
print("WINDOWING TEST PASSED: out-of-window -> CLS(0,0); in-window -> correct token span.")
