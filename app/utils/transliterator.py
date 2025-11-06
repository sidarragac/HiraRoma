class Transliterator:
    def __init__(self):
        self.doubles = {
            'aa': 'a',
            'ii': 'i',
            'uu': 'u',
            'ee': 'e',
            'oo': 'o',
            'nn': 'n'
        }
        self.equivalences = {
            'a': 'あ',
            'i': 'い',
            'u': 'う',
            'e': 'え',
            'o': 'お',

            # K
            'ka': 'か',
            'ki': 'き',
            'ku': 'く',
            'ke': 'け',
            'ko': 'こ',

            # S
            'sa': 'さ',
            'shi': 'し',
            'su': 'す',
            'se': 'せ',
            'so': 'そ',

            # T
            'ta': 'た',
            'chi': 'ち',
            'tsu': 'つ',
            'te': 'て',
            'to': 'と',

            # N
            'na': 'な',
            'ni': 'に',
            'nu': 'ぬ',
            'ne': 'ね',
            'no': 'の',

            # H
            'ha': 'は',
            'hi': 'ひ',
            'fu': 'ふ',
            'he': 'へ',
            'ho': 'ほ',

            # M
            'ma': 'ま',
            'mi': 'み',
            'mu': 'む',
            'me': 'め',
            'mo': 'も',

            # Y
            'ya': 'や',
            'yu': 'ゆ',
            'yo': 'よ',

            # R
            'ra': 'ら',
            'ri': 'り',
            'ru': 'る',
            're': 'れ',
            'ro': 'ろ',

            # W
            'wa': 'わ',
            'wo': 'を',

            # N
            'n': 'ん',
        }

    def transliterate(self, char):
        roma = self.equivalences.get(char)
        if roma:
            return (char, roma)
        
        single_char = self.doubles.get(char, '') # Default '', but will never reach that condition.
        roma = self.equivalences.get(single_char)

        return (single_char, roma)

    def transliterate_text(self, text):
        roma_text = []
        for char in text:
            roma = self.transliterate(char)
            roma_text.append(roma)

        return roma_text
