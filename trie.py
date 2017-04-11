class Trie:
    """
    Implement a trie.

    Example:
    import trie
    iterable = ['apple', 'and', 'boy']
    tr = trie.Trie(iterable)
    s = 'That boy is eating an apple'

    #First way for string matching:
    print("\t".join(tr.maxmatchall(s)))

    #Second way for string matching, 
    #where you can access the index of matched string.
    current_position = 0
    while(current_index < len(s)):
        add_index = tr.maxmatch(s[current_index:])
        if add_index == 0:
            current_index += 1
        else:
            print(s[current_index:current_index + add_index])
            current_index += add_index
    """
    def __init__(self, words):
        self._tree = {}
        for word in words:
            self.insert(word)

    def insert(self, word):
        current = self._tree
        for character in word:
            current = current.setdefault(character, {})
        current.setdefault("_end")

    def search(self, word):
        current = self._tree
        for character in word:
            if character not in current:
                return False
            current = current[character]
        if "_end" in current:
            return True
        return False

    def maxmatch(self, string):
        current = self._tree
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

    @property
    def tree(self):
        return self._tree
