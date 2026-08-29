from mlbricks import eon
assert callable(eon)
import torch


def test_fast_generation_preserves_state_dict_and_matches_full_forward():
    torch.manual_seed(7)
    model = eon(dim=16, width=24, depth=2, mixer='esa', ffn='saffn', mixer_config={'head': 4}, backend='pytorch', precision='fp32').eval()
    keys = list(model.state_dict())
    prompt = torch.randn(2, 5, 16)
    full = model(prompt)
    prefill, cache = model.prefill(prompt)
    torch.testing.assert_close(prefill, full, rtol=0, atol=0)
    model.prepare_generation()
    assert list(model.state_dict()) == keys
    prefix = prompt
    for _ in range(3):
        token = torch.randn(2, 1, 16)
        step, cache = model.decode_step(token, cache)
        prefix = torch.cat([prefix, token], dim=1)
        reference = model(prefix)[:, -1:]
        torch.testing.assert_close(step, reference, rtol=1e-5, atol=1e-5)
