def warmup_linear(x, warmup=0.002):
    if warmup == 0.0:
        return 1.0
    elif x < warmup:
        return x/warmup
    return 1.0 - x

def warmup_linear_decay_exp(global_step, decay_rate, decay_steps, total_steps, warmup=0.002):
    x = global_step/total_steps
    warmup_end = warmup * total_steps
    if warmup == 0.0:
        return 1.0
    elif x < warmup:
        return x/warmup
    return decay_rate**((global_step-warmup_end)/decay_steps)