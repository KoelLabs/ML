import panphon
import panphon.distance
from fastdtw import fastdtw
import numpy as np
from scipy.spatial.distance import euclidean

# Create a panphon feature table
ft = panphon.FeatureTable()


# Convert a phoneme to a numerical feature vector
def phoneme_to_vector(phoneme):
    vectors = ft.word_to_vector_list(phoneme, numeric=True)
    if vectors:
        return np.array(vectors[0])  # Take the first vector if multiple exist
    else:
        return None  # Invalid phoneme


# Convert sequences of phonemes to sequences of vectors
def sequence_to_vectors(seq):
    return [phoneme_to_vector(p) for p in seq if phoneme_to_vector(p) is not None]


def fastdtw_phoneme_alignment(seq1, seq2):
    # Convert phoneme sequences to feature vector sequences
    seq1_vectors = sequence_to_vectors(seq1)
    seq2_vectors = sequence_to_vectors(seq2)

    if not seq1_vectors or not seq2_vectors:
        raise ValueError(
            "One or both sequences could not be converted to feature vectors."
        )

    # Use FastDTW with Euclidean distance on the vectors
    distance, path = fastdtw(seq1_vectors, seq2_vectors, dist=euclidean)

    # Align the original phoneme sequences based on the path
    aligned_seq1 = []
    aligned_seq2 = []
    for i, j in path:
        aligned_seq1.append(seq1[i] if i < len(seq1) else "-")
        aligned_seq2.append(seq2[j] if j < len(seq2) else "-")

    return "".join(aligned_seq1), "".join(aligned_seq2)


def pair_by_words(target, target_by_words, speech):
    paired = zip(*fastdtw_phoneme_alignment(target, speech))

    pair_by_words = []
    pairs = iter(paired)
    cur_pair = next(pairs)
    start = []
    for word, phons in target_by_words:
        phons = list(phons)
        ps = start
        while len(phons) > 0:
            t, s = cur_pair
            if t != phons[0]:
                phons.pop(0)
            ps.append(cur_pair)
            try:
                cur_pair = next(pairs)
            except StopIteration:
                break
        pair_by_words.append((word, ps[:-1]))
        start = [ps[-1]]

    return pair_by_words


def score_words_cer(target, target_by_words, speech):
    pbw = pair_by_words(target, target_by_words, speech)
    word_scores = []
    average_score = 0
    for word, pairs in pbw:
        cer = sum(1 for t, s in pairs if t != s) / len(pairs)
        seq1, seq2 = map(lambda x: "".join(x), zip(*pairs))
        word_scores.append((word, seq1, seq2, (1 - cer / 2)))
        average_score += 1 - cer / 2
    average_score /= len(pbw)
    return word_scores, average_score


def score_words_wfed(target, target_by_words, speech):
    pbw = pair_by_words(target, target_by_words, speech)
    word_scores = []
    average_score = 0
    for word, pairs in pbw:
        seq1, seq2 = map(lambda x: "".join(x), zip(*pairs))
        norm_score = (
            22 - panphon.distance.Distance().weighted_feature_edit_distance(seq1, seq2)
        ) / 22
        word_scores.append((word, seq1, seq2, norm_score**2))
        average_score += norm_score**2
    average_score /= len(pbw)
    return word_scores, average_score


def feedback(target, target_by_words, speech, good_enough_threshold=0.4):
    pbw = pair_by_words(target, target_by_words, speech)
    word_feedbacks = []
    for word, pairs in pbw:
        wrongest_pair = pairs[0]
        wrongest_pair_dist = panphon.distance.Distance().weighted_feature_edit_distance(
            wrongest_pair[0], wrongest_pair[1]
        )
        for p in pairs:
            dist = panphon.distance.Distance().weighted_feature_edit_distance(
                p[0], p[1]
            )
            if dist > wrongest_pair_dist:
                wrongest_pair = p
                wrongest_pair_dist = dist
        if wrongest_pair_dist < good_enough_threshold:
            word_feedbacks.append(
                (
                    word,
                    'Your pronunciation of "' + word + '" is perfect!',
                    wrongest_pair_dist,
                )
            )
        else:
            target, speech = wrongest_pair
            t, s = sound_descriptions[target], sound_descriptions[speech]
            word_feedbacks.append(
                (
                    word,
                    f"""
                The actor made the '{t['phonemicSpelling']}' sound in "{word}" but you made the '{s['phonemicSpelling']}' sound.
                It is supposed to be {t['description'][0].lower() + t['description'][1:]}
                {t["exampleWord"]}
                """.strip(),
                    wrongest_pair_dist,
                )
            )
    top3 = sorted(word_feedbacks, key=lambda x: x[2], reverse=True)[:3]
    return word_feedbacks, top3


def side_by_side_description(target, target_by_words, speech):
    pbw = pair_by_words(target, target_by_words, speech)
    vis = []
    prevt, prevs = target[1], speech[1]
    for word, pairs in pbw:
        st = []
        for t, s in pairs:
            vt, vs = (
                sound_descriptions[t]["viseme_id"]
                or sound_descriptions[prevt]["viseme_id"],
                sound_descriptions[s]["viseme_id"]
                or sound_descriptions[prevs]["viseme_id"],
            )
            st.append(
                (
                    {**sound_descriptions[t], "viseme_id": vt[0]},
                    {**sound_descriptions[s], "viseme_id": vs[0]},
                )
            )
            prevt, prevs = t, s
        vis.append((word, st))
    vis[0] = (
        vis[0][0],
        [
            (
                {
                    "phonemicSpelling": "silence",
                    "description": "Mouth closed.",
                    "exampleWord": "",
                    "example_words": ["silence"],
                    "viseme_id": 0,
                },
                {
                    "phonemicSpelling": "silence",
                    "description": "Mouth closed.",
                    "exampleWord": "",
                    "example_words": ["silence"],
                    "viseme_id": 0,
                },
            ),
            *vis[0][1],
        ],
    )
    return vis


sound_descriptions = {
    "a": {
        "phonemicSpelling": "ah",
        "description": "An open front unrounded vowel. Open your mouth wide, position the tongue low and towards the front, and vibrate the vocal cords.",
        "exampleWord": "This is the vowel sound in 'father'.",
        "viseme_id": [9],
        "example_words": ["*ou*t", "s*ou*th", "c*ow*"],
    },
    "b": {
        "phonemicSpelling": "buh",
        "description": "A voiced bilabial stop. Press both lips together, then release while vibrating the vocal cords.",
        "exampleWord": "This is the initial sound in 'bat'.",
        "viseme_id": [21],
        "example_words": ["*b*ig", "num*b*er", "cra*b*"],
    },
    "d": {
        "phonemicSpelling": "duh",
        "description": "A voiced alveolar stop. Place the tongue against the alveolar ridge, stop airflow, then release while vibrating the vocal cords.",
        "exampleWord": "This is the initial sound in 'dog'.",
        "viseme_id": [19],
        "example_words": ["*d*ig", "ran*d*om", "ro*d*"],
    },
    "e": {
        "phonemicSpelling": "ay",
        "description": "A close-mid front unrounded vowel. Keep the tongue mid-high and towards the front, and vibrate the vocal cords.",
        "exampleWord": "This is the vowel sound in 'say' (in non-rhotic accents).",
        "viseme_id": [4],
        "example_words": ["*e*very", "p*e*t", "m*eh* (rare word-final)"],
    },
    "f": {
        "phonemicSpelling": "fuh",
        "description": "A voiceless labiodental fricative. Place the upper teeth against the lower lip and push air through without vibrating the vocal cords.",
        "exampleWord": "This is the initial sound in 'fish'.",
        "viseme_id": [18],
        "example_words": ["*f*ork", "le*f*t", "hal*f*"],
    },
    "h": {
        "phonemicSpelling": "huh",
        "description": "A voiceless glottal fricative. Push air through the open vocal cords without vibrating them.",
        "exampleWord": "This is the initial sound in 'hat'.",
        "viseme_id": [12],
        "example_words": ["*h*elp", "en*h*ance", "a-*h*a!"],
    },
    "i": {
        "phonemicSpelling": "ee",
        "description": "A close front unrounded vowel. Raise the tongue high and towards the front, and vibrate the vocal cords.",
        "exampleWord": "This is the vowel sound in 'see'.",
        "viseme_id": [6],
        "example_words": ["*ea*t", "f*ee*l", "vall*ey*"],
    },
    "j": {
        "phonemicSpelling": "yuh",
        "description": "A voiced palatal approximant. Place the tongue close to the hard palate without touching, and vibrate the vocal cords.",
        "exampleWord": "This is the initial sound in 'yes'.",
        "viseme_id": [6],
        "example_words": ["*y*ard", "f*e*w", "on*i*on"],
    },
    "k": {
        "phonemicSpelling": "kuh",
        "description": "A voiceless velar stop. Place the back of the tongue against the soft palate, stop airflow, then release without vibrating the vocal cords.",
        "exampleWord": "This is the initial sound in 'cat'.",
        "viseme_id": [20],
        "example_words": ["*c*ut", "sla*ck*er", "Iraq"],
    },
    "l": {
        "phonemicSpelling": "luh",
        "description": "A voiced alveolar lateral approximant. Place the tongue against the alveolar ridge, allowing air to pass along the sides while vibrating the vocal cords.",
        "exampleWord": "This is the initial sound in 'leaf'.",
        "viseme_id": [14],
        "example_words": ["*l*id", "g*l*ad", "pa*l*ace", "chi*ll*"],
    },
    "m": {
        "phonemicSpelling": "muh",
        "description": "A voiced bilabial nasal. Press both lips together, and let air pass through the nose while vibrating the vocal cords.",
        "exampleWord": "This is the initial sound in 'man'.",
        "viseme_id": [21],
        "example_words": ["*m*at", "s*m*ash", "ca*m*era", "roo*m*"],
    },
    "n": {
        "phonemicSpelling": "nuh",
        "description": "A voiced alveolar nasal. Place the tongue against the alveolar ridge, and let air pass through the nose while vibrating the vocal cords.",
        "exampleWord": "This is the initial sound in 'net'.",
        "viseme_id": [19],
        "example_words": ["*n*o", "s*n*ow", "te*n*t", "chicke*n*"],
    },
    "o": {
        "phonemicSpelling": "oh",
        "description": "A close-mid back rounded vowel. Round the lips, keep the tongue mid-high and towards the back, and vibrate the vocal cords.",
        "exampleWord": "This is the vowel sound in 'go' (in non-rhotic accents).",
        "viseme_id": [8],
        "example_words": ["*o*ld", "cl*o*ne", "g*o*"],
    },
    "p": {
        "phonemicSpelling": "puh",
        "description": "A voiceless bilabial stop. Press both lips together, then release without vibrating the vocal cords.",
        "exampleWord": "This is the initial sound in 'pat'.",
        "viseme_id": [21],
        "example_words": ["*p*ut", "ha*pp*en", "fla*p*"],
    },
    "s": {
        "phonemicSpelling": "sss",
        "description": "A voiceless alveolar fricative. Place the tongue near the alveolar ridge, and push air through without vibrating the vocal cords.",
        "exampleWord": "This is the initial sound in 'sit'.",
        "viseme_id": [15],
        "example_words": ["*s*it", "ri*s*k", "fact*s*"],
    },
    "t": {
        "phonemicSpelling": "tuh",
        "description": "A voiceless alveolar stop. Place the tongue against the alveolar ridge, stop airflow, then release without vibrating the vocal cords.",
        "exampleWord": "This is the initial sound in 'top'.",
        "viseme_id": [19],
        "example_words": ["*t*alk", "capi*t*al", "sough*t*"],
    },
    "u": {
        "phonemicSpelling": "oo",
        "description": "A close back rounded vowel. Round the lips, keep the tongue high and towards the back, and vibrate the vocal cords.",
        "exampleWord": "This is the vowel sound in 'blue'.",
        "viseme_id": [7],
        "example_words": ["*U*ber", "b*oo*st", "t*oo*"],
    },
    "v": {
        "phonemicSpelling": "vuh",
        "description": "A voiced labiodental fricative. Place the upper teeth against the lower lip, push air through, and vibrate the vocal cords.",
        "exampleWord": "This is the initial sound in 'van'.",
        "viseme_id": [18],
        "example_words": ["*v*alue", "e*v*ent", "lo*v*e"],
    },
    "w": {
        "phonemicSpelling": "wuh",
        "description": "A voiced labio-velar approximant. Round the lips and raise the back of the tongue towards the soft palate while vibrating the vocal cords.",
        "exampleWord": "This is the initial sound in 'win'.",
        "viseme_id": [7],
        "example_words": ["*w*ith", "s*ue*de", "al*w*ays"],
    },
    "z": {
        "phonemicSpelling": "zzz",
        "description": "A voiced alveolar fricative. Place the tongue near the alveolar ridge, push air through, and vibrate the vocal cords.",
        "exampleWord": "This is the initial sound in 'zebra'.",
        "viseme_id": [15],
        "example_words": ["*z*ap", "bu*s*y", "kid*s*"],
    },
    "æ": {
        "phonemicSpelling": "ah",
        "description": "A near-open front unrounded vowel. Open your mouth widely and position the tongue low and towards the front.",
        "exampleWord": "This is the vowel sound in 'cat'.",
        "viseme_id": [1],
        "example_words": ["*a*ctive", "c*a*t", "n*ah* (rare word-final)"],
    },
    "ð": {
        "phonemicSpelling": "th",
        "description": "A voiced dental fricative. Place the tongue between the teeth, push air through, and vibrate the vocal cords.",
        "exampleWord": "This is the initial sound in 'this'.",
        "viseme_id": [17],
        "example_words": ["*th*en", "mo*th*er", "smoo*th*"],
    },
    "ŋ": {
        "phonemicSpelling": "ng",
        "description": "A voiced velar nasal. Place the back of the tongue against the soft palate, and let air pass through the nose while vibrating the vocal cords.",
        "exampleWord": "This is the final sound in 'sing'.",
        "viseme_id": [20],
        "example_words": ["li*n*k", "si*ng*"],
    },
    "ɑ": {
        "phonemicSpelling": "ah",
        "description": "An open back unrounded vowel. Open your mouth wide, position the tongue low and towards the back, and vibrate the vocal cords.",
        "exampleWord": "This is the vowel sound in 'spa' (in non-rhotic accents).",
        "viseme_id": [2],
        "example_words": ["*o*bstinate", "p*o*ppy", "r*ah*", "(rare word-final)"],
    },
    "ɔ": {
        "phonemicSpelling": "aw",
        "description": "An open-mid back rounded vowel. Round the lips and lower the tongue towards the back.",
        "exampleWord": "This is the vowel sound in 'thought' (in non-rhotic accents).",
        "viseme_id": [3],
        "example_words": ["*o*range", "c*au*se", "Ut*ah*"],
    },
    "ə": {
        "phonemicSpelling": "uh",
        "description": "A mid-central unrounded vowel. Keep the tongue relaxed and central, and vibrate the vocal cords.",
        "exampleWord": "This is the vowel sound in the first syllable of 'about'.",
        "viseme_id": [1],
        "example_words": ["*a*go", "wom*a*n", "are*a*"],
    },
    "ɛ": {
        "phonemicSpelling": "eh",
        "description": "An open-mid front unrounded vowel. Lower the tongue slightly towards the front.",
        "exampleWord": "This is the vowel sound in 'bed'.",
        "viseme_id": [4],
        "example_words": ["*e*very", "p*e*t", "m*eh* (rare word-final)"],
    },
    "ɡ": {
        "phonemicSpelling": "guh",
        "description": "A voiced velar stop. Place the back of the tongue against the soft palate, stop airflow, then release while vibrating the vocal cords.",
        "exampleWord": "This is the initial sound in 'go'.",
        "viseme_id": [20],
        "example_words": ["*g*o", "a*g*o", "dra*g*"],
    },
    "ɪ": {
        "phonemicSpelling": "ih",
        "description": "A near-close front unrounded vowel. Raise the tongue high and towards the front, but not as close as /i/.",
        "exampleWord": "This is the vowel sound in 'sit'.",
        "viseme_id": [6],
        "example_words": ["*i*f", "f*i*ll"],
    },
    "ɹ": {
        "phonemicSpelling": "ruh",
        "description": "A voiced alveolar approximant. Curl the tongue towards the alveolar ridge without touching, and vibrate the vocal cords.",
        "exampleWord": "This is the initial sound in 'red'.",
        "viseme_id": [13],
        "example_words": ["*r*ed", "b*r*ing", "bo*rr*ow", "ta*r*"],
    },
    "ɾ": {
        "phonemicSpelling": "flap",
        "description": "A voiced alveolar tap. Quickly tap the tongue against the alveolar ridge.",
        "exampleWord": "This is the middle sound in 'butter' (in American English).",
        "viseme_id": [19],
        "example_words": ["mu*dd*y", "wi*tt*y", "fla*tt*ers", "a*dd*"],
    },
    "ʃ": {
        "phonemicSpelling": "sh",
        "description": "A voiceless postalveolar fricative. Place the tongue near the roof of the mouth, just behind the alveolar ridge, and push air through.",
        "exampleWord": "This is the initial sound in 'shoe'.",
        "viseme_id": [16],
        "example_words": ["*sh*e", "abbrevia*ti*on", "ru*sh*"],
    },
    "ʊ": {
        "phonemicSpelling": "uh",
        "description": "A near-close back rounded vowel. Round the lips and raise the tongue towards the back.",
        "exampleWord": "This is the vowel sound in 'put'.",
        "viseme_id": [4],
        "example_words": ["b*oo*k"],
    },
    "ʌ": {
        "phonemicSpelling": "uh",
        "description": "An open-mid back unrounded vowel. Lower the tongue towards the back and open the mouth slightly.",
        "exampleWord": "This is the vowel sound in 'cup'.",
        "viseme_id": [1],
        "example_words": ["*u*ncle", "c*u*t"],
    },
    "ʒ": {
        "phonemicSpelling": "zh",
        "description": "A voiced postalveolar fricative. Place the tongue near the roof of the mouth, just behind the alveolar ridge, and push air through while vibrating the vocal cords.",
        "exampleWord": "This is the middle sound in 'measure'.",
        "viseme_id": [16],
        "example_words": ["*J*acques", "plea*s*ure", "gara*g*e"],
    },
    "ʔ": {
        "phonemicSpelling": "glottal stop",
        "description": "A voiceless glottal stop. Close the vocal cords briefly, then release to produce a stop sound.",
        "exampleWord": "This is the catch in the middle of 'uh-oh'.",
        "viseme_id": [],
        "example_words": ["butt*o*n", "kitt*e*n"],
    },
    "θ": {
        "phonemicSpelling": "th",
        "description": "A voiceless dental fricative. Place the tongue between the teeth and push air through without vibrating the vocal cords.",
        "exampleWord": "This is the initial sound in 'think'.",
        "viseme_id": [19],
        "example_words": ["*th*in", "empa*th*y", "mon*th*"],
    },
}
