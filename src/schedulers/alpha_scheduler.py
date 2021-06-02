import math


class StaticAlphaScheduler:
    def __init__(self, max_alpha):
        self.alpha = max_alpha

    def step(self):
        pass

class LinearAlphaScheduler:
    def __init__(self, max_alpha, epoch):
        self.max_alpha = max_alpha
        self.epoch = epoch

        self.alpha = 0.

        self.last_step = -1
        
    def step(self):
        self.last_step += 1

        a = min(1., (self.last_step / self.epoch))
        self.alpha = a * self.max_alpha

class CosAlphaScheduler:
    def __init__(self, max_alpha, cycle):
        self.max_alpha = max_alpha
        self.cycle = cycle

        self.alpha = 0.

        self.last_step = -1
        
    def step(self):
        self.last_step += 1

        if self.last_step == self.cycle:
            self.last_step = 0
        
        
        a = self.last_step / (self.cycle-1)
        ang = a * (math.pi / 2)
        
        self.alpha = self.max_alpha - math.cos(ang) * self.max_alpha

if __name__ == "__main__":
    alpha = 5000
    static = StaticAlphaScheduler(alpha)
    linear = LinearAlphaScheduler(alpha, 300)
    cos = CosAlphaScheduler(alpha, 50)

    for _ in range(500):
        static.step()
        linear.step()
        cos.step()
        print(f"{round(static.alpha):>10} | {round(linear.alpha):>10} | {round(cos.alpha):>10}")

