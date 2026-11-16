>>> import torch
>>> from fla.layers import MultiScaleRetention
>>> batch_size, num_heads, seq_len, hidden_size = 32, 4, 2048, 1024
>>> device, dtype = 'cuda:0', torch.bfloat16
>>> retnet = MultiScaleRetention(hidden_size=hidden_size, num_heads=num_heads).to(device=device, dtype=dtype)
>>> retnet
MultiScaleRetention(
  (q_proj): Linear(in_features=1024, out_features=1024, bias=False)
  (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
  (v_proj): Linear(in_features=1024, out_features=2048, bias=False)
  (g_proj): Linear(in_features=1024, out_features=2048, bias=False)
  (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
  (g_norm_swish_gate): FusedRMSNormGated(512, eps=1e-05, activation=swish)
  (rotary): RotaryEmbedding(dim=256, base=10000.0, interleaved=False, pos_idx_in_fp32=True)
)
>>> x = torch.randn(batch_size, seq_len, hidden_size).to(device=device, dtype=dtype)
>>> y, *_ = retnet(x)
>>> y.shape
torch.Size([32, 2048, 1024])
