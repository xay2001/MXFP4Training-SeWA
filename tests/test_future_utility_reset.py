from types import SimpleNamespace

import torch

from quantization.QLinear import QLinearLayer


def make_args():
    return SimpleNamespace(
        qchoice=['all'],
        fwbit=4,
        bwbit=4,
        fabit=4,
        babit=4,
        fwexp=2,
        bwexp=2,
        faexp=2,
        baexp=2,
        symm=True,
        row_blocksize=1,
        column_blocksize=32,
        epsilon=1e-8,
        tritonQ=False,
        mxscale=1,
        qlinear_f_w_in=False,
        qlinear_f_a_in=False,
        qlinear_b_w_rq=False,
        qlinear_b_a_rq=False,
        qlinear_b_dy=False,
        qlinear_b_dy_t=False,
        qlinear_all=True,
        qlinear_ema_decay=0.9,
        qlinear_osmq=False,
        qlinear_future_utility_reset=True,
        qlinear_future_utility_start_step=0,
        qlinear_future_reset_interval=5,
        qlinear_future_candidate_ratio=0.25,
        qlinear_future_budget_ratio=0.125,
        qlinear_future_utility_decay=0.9,
        qlinear_future_flip_decay=0.9,
        qlinear_future_utility_tau=0.1,
    )


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(7)
    layer = QLinearLayer(64, 64, bias=False, args=make_args(), layer_type='mlp_fc1').cuda()
    layer.layer_name = 'dfur_smoke'
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.05)

    decision_stats = []
    for step in range(20):
        if step >= 10:
            with torch.no_grad():
                layer.weight.add_(0.02 * torch.randn_like(layer.weight))

        x = torch.randn(32, 64, device='cuda')
        target = torch.randn(32, 64, device='cuda')
        optimizer.zero_grad()
        loss = (layer(x) - target).square().mean()
        loss.backward()
        optimizer.step()

        if step == 14:
            with torch.no_grad():
                layer.fur_utility_ema.fill_(0.5)

        stats = layer.update_future_utility_reset(step)
        if stats is not None:
            decision_stats.append(stats)

        with torch.no_grad():
            layer.ema_step.add_(1)
            layer.ema_weight.mul_(layer.ema_decay).add_(layer.weight, alpha=1.0 - layer.ema_decay)

        assert torch.isfinite(loss)
        assert torch.isfinite(layer.weight).all()
        assert torch.isfinite(layer.fur_utility_ema).all()

    assert int(layer.fur_utility_updates.item()) > 0
    assert decision_stats
    assert any(item['candidate_count'] > 0 for item in decision_stats)
    assert any(item['reset_count'] > 0 for item in decision_stats)
    print({'loss': float(loss.item()), 'decisions': decision_stats})


if __name__ == '__main__':
    main()
