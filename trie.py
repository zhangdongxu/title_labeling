class Trie:
    """
    Implement a trie.
    Insert.
    Search.
    maxmatch.
    """
    def __init__(self, tree = {}):
        self.tree = tree

    def insert(self, word):
        current = self.tree
        for character in word:
            current = current.setdefault(character, {})
        current.setdefault("_end")

    def search(self, word):
        current = self.tree
        for character in word:
            if character not in current:
                return False
            current = current[character]
        if "_end" in current:
            return True
        return False

    def maxmatch(self, string):
        current = self.tree
        current_position = 0
        matched_last_position = 0
        for character in string:
            if "_end" in current:
                matched_last_position = current_position
            if character not in current:
                return matched_last_position
            else:
                current_position += 1
                current = current[character]
        if "_end" in current:
            matched_last_position = current_position
        return matched_last_position 

    def maxmatchall(self, string):
        matched_words = []
        history_position = 0
        while(history_position < len(string)):
            offset = self.maxmatch(string[history_position:])
            if offset == 0:
                offset = 1
            else:
                matched_words.append(string[history_position:history_position + offset])
            history_position += offset
        return matched_words

def load(words):
    trie = Trie()
    for word in words:
        trie.insert(word)
    return trie
