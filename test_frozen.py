import numpy as np, torch, run_experiments as R

# 1) FrozenHead forward: LSTM config and linear-only config
for lstm in ([8], []):
    head = R.build_frozen_head({"lstm": lstm}, hidden=16)
    emb = torch.randn(4, 5, 16)
    sl, el = head(emb)
    assert sl.shape == (4, 5) and el.shape == (4, 5), (lstm, sl.shape)
print("FrozenHead forward OK (LSTM + linear-only), logits shape [B, L]")

# 2) encode_split with a mock frozen encoder -> [N, L, H] float16
class MockEnc:
    def eval(self): return self
    def __call__(self, input_ids=None, attention_mask=None):
        b, L = input_ids.shape
        class O: pass
        o = O(); o.last_hidden_state = torch.randn(b, L, 16)
        return o
feats = [dict(input_ids=[1,2,3,4,5], attention_mask=[1,1,1,1,1], start=1, end=3) for _ in range(7)]
emb = R.encode_split(MockEnc(), feats, device="cpu", bs=3)
assert emb.shape == (7, 5, 16) and emb.dtype == np.float16, (emb.shape, emb.dtype)
print("encode_split OK -> shape", emb.shape, emb.dtype)

# 3) one training step on cached embeddings (loss finite, params update)
head = R.build_frozen_head({"lstm": [8]}, hidden=16)
opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
p0 = next(head.parameters()).clone()
R.train_head_epoch(head, emb, feats, opt, "cpu", bs=3, smoke=False)
assert not torch.equal(p0, next(head.parameters())), "params did not update"
vl = R.eval_head_loss(head, emb, feats, "cpu", bs=3, smoke=False)
assert np.isfinite(vl), vl
print("train_head_epoch + eval_head_loss OK (params updated, val loss finite)")

# 4) frozen eval end-to-end on a tiny RE-120-like example
ctx = "compress large image files"
words = [(m.start(), m.end()) for m in __import__("re").finditer(r"\S+", ctx)]
offs = [None] + [w for w in words] + [None]
seq  = [0] + [1]*len(words) + [0]
ef = [dict(input_ids=[0]*(len(words)+2), attention_mask=[1]*(len(words)+2),
           offsets=offs, seq_ids=seq, sample=0)]
eval_emb = np.random.randn(1, len(words)+2, 16).astype(np.float16)
re120 = {"context": [ctx], "gold": ["large image"]}
em, f1, rg = R.evaluate_re120_frozen(head, eval_emb, ef, re120, "cpu", bs=4)
assert 0.0 <= em <= 1 and 0.0 <= f1 <= 1 and 0.0 <= rg <= 1, (em, f1, rg)
print(f"evaluate_re120_frozen OK (EM={em}, F1={f1:.2f}, ROUGE={rg:.2f})")
print("\nALL FROZEN-PATH TESTS PASSED")
