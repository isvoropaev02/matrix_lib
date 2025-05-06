class TextBlock:
    def __init__(self, rows):
        assert isinstance(rows, list)
        self.rows = rows
        self.height = len(self.rows)
        self.width = max(map(len, self.rows))

    @classmethod
    def from_str(_cls, data):
        assert isinstance(data, str)
        return TextBlock(data.split("\n"))

    def format(self, width=None, height=None):
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        return [f"{row:{width}}" for row in self.rows] + [" " * width] * (
            height - self.height
        )

    @staticmethod
    def merge(blocks):
        return [" ".join(row) for row in zip(*blocks)]
