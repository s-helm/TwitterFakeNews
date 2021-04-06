class Counter:

    def __init__(self):
        self.count = 0

    def reset_count(self):
        self.count = 0

    def increase_count(self):
        self.count += 1

    def get_count(self):
        return self.count
