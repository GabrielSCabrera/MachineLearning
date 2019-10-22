def gen(low, high, step = 1):
    low = low - step
    while low < high-step:
        low += step
        yield low

for i in gen(0, 10, 2):
    print(i)
