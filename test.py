
__target_token_index = {
    '\t': 0, '\n': 1, ' ': 2, '!': 3, '"': 4,
    '(': 5, ')': 6, '-': 7, '0': 8, '1': 9,
            '2': 10, '3': 11, '4': 12, '5': 13,
            '6': 14, '7': 15, '8': 16, '9': 17, 'a': 18,
            'b': 19, 'c': 20, 'd': 21, 'e': 22, 'f': 23,
            'g': 24, 'h': 25, 'i': 26, 'k': 27, 'l': 28,
            'm': 29, 'n': 30, 'o': 31, 'p': 32, 'r': 33,
            's': 34, 't': 35, 'u': 36, 'w': 37, 'y': 38
}

__reverse_target_char_index = dict(
    (i, char) for char, i in __target_token_index.items()
)

print(__reverse_target_char_index)