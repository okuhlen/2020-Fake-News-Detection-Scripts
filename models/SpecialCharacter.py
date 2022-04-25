class SpecialCharacter:

    def __init__(self, symbol, description, id):
        self.symbol = symbol
        self.description = description
        self.id = id

    def get_symbol(self):
        return self.symbol

    def get_description(self):
        return self.description

    def get_id(self):
        return self.id