import time


class RateLimitCounter:
    def __init__(self, rate_limit: int, period: int):
        self.rate_limit = rate_limit
        self.period = period
        self.count = 0
        self.last_reset = time.time()

    def is_first_request(self):
        return self.count == 0

    def increment(self):
        now = time.time()
        if now - self.last_reset > self.period:
            self.count = 0
            self.last_reset = now
        self.count += 1
        return self.count <= self.rate_limit

    def reset(self):
        self.count = 0
        self.last_reset = time.time()

    def get_count(self):
        return self.count

    def get_rate_limit(self):
        return self.rate_limit

    def get_period(self):
        return self.period
