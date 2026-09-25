class EarlyStopping:
    def __init__(self, patience: int = 30, delta: float = 0.0, verbose: bool = False) -> None:
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.reset()

    def reset(self) -> None:
        self.best_score = None
        self.counter = 0

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
