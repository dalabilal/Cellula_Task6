class AgentMemory:
    def __init__(self, owner: str):
        self.owner = owner
        self._items: list[str] = []

    def _check(self, caller: str):
        if caller != self.owner:
            raise PermissionError(f"'{caller}' is not allowed to access the {self.owner} memory")

    def add(self, caller: str, text: str):
        self._check(caller)
        self._items.append(text)

    def recent(self, caller: str, n: int = 6) -> str:
        self._check(caller)
        return "\n".join(self._items[-n:]) or "none"

    def clear(self, caller: str):
        self._check(caller)
        self._items.clear()
