import jphones as j2p
text = '一 これ は 、 わたし が 小さい とき に 、'
# strip spaces and commas out of text
text = text.replace(" ", "").replace(",", "")

token = {'token': text, 'type': 'word'}

Phonetizer = j2p.phonetizer.Phonetizer()
phonemes = Phonetizer.get_phonemes(token)

print(phonemes)
# {'phonemes': ['s', 'u', 'g', 'o', 'i'], 'token': 'すごい', 'type': 'word'}
