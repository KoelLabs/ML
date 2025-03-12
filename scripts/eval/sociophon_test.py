import os
import sys

from metrics import per, fer

# Given transcript
# given_transcript = "p l iː s k oʊ l s t ɛ l ɚ ɑː s k ɚ t ə b ɹ ɪ ŋ ð iː z θ ɪ ŋ z w ɪ v h ɚ f ɹ ʌ m ð ə s t oʊ s ɪ k s p uː n z ʌ v f ɹ ɪ s n oʊ p iː s f aɪ v θ ɪ k s l æ p s ʌ v b l uː tʃ iː z æ n d m eɪ b i ɐ s n æ k f ɔːɹ h ɜː b ɹ ʌ ð ɚ ɚ b ɹ æ ð ɚ b ɔ w iː ɔː l s oʊ n iː d ɐ s m oʊ l p l æ s t ɪ k s n eɪ k æ n d ɐ b ɪ ɡ t uː ɐ f ɹ ɔ k f ɔːɹ ð ə k ɪ d z ʃ iː k æ n s k uː p ð iː z θ ɪ ŋ z ɪ n t ʊ f ɹ iː ɹ ɪ t b æ k s æ n w iː w ɪ l ɡ oʊ m iː ɾ ɐ w ɪ n s t eɪ æ t ə d ɹ eɪ n s t eɪ ʃ ə n"
# given_transcript = "p l iː z k ɔː l s t ɛ l ɐ ɛ s k ɚ ɾ ə b ɹ ɪ ŋ ð oʊ z θ ɪ ŋ z w ɪ ð h ɚ f ɹ ʌ m ð ə s t oːɹ s ɪ k s p uː n z ʌ v f ɹ ɛ ʃ s n oʊ p iː z f aɪ v θ ɪ k s l æ b z ʌ v b l uː tʃ iː z æ n d m eɪ b i ɐ s n æ k f ɔːɹ h ɚ b ɹ ʌ ð ɚ b ɹ ɔː b ɑː b w iː ɔː l s oʊ n iː d ɐ s m ɔː l p l eɪ s t ɪ k s n eɪ k æ n d ɐ b ɪ ɡ t ɔːɹ f ɹ ɔː ɡ f ɔːɹ ð ə k ɪ d z ʃ iː k æ n s k uː p ð iː s θ ɪ ŋ z ɪ n t ə θ ɹ iː ɹ ɛ d b æ ɡ z æ n d w iː w əl ɡ oʊ m iː ɾ h ɚ w ɛ n z d eɪ æ t ð ə t ɹ eɪ n s t eɪ ʃ ə n"
# given_transcript = "p l iː z k oʊ l s t ɛ l ɐ ɔː s k ɚ t ə b ɹ ɪ ŋ ð iː z θ ɪ ŋ z w ɪ θ ɚ f ɹ ʌ m ð ə s t oːɹ s ɪ k s p uː n z ʌ v f ɹ ɛ ʃ s n aʊ p iː z f aɪ v θ ɪ k s l æ b z ʌ v b l uː tʃ iː z æ n d m eɪ b i ɐ s n æ k f ɔːɹ h ɚ b ɹ ʌ ð ɚ b ɑː b w iː ɔː l s oʊ n iː d ɐ s m oʊ l p l æ s t ɪ k s n aɪ k æ n d ɐ b ɪ ɡ t oːɹ f ɹ ɔ ɡ f ɔːɹ ð ə k ɪ d z ʃ iː k æ n s k uː p ð iː z θ ɪ ŋ z ɪ n t ʊ θ ɹ iː ɹ iː d b æ ɡ z æ n d w iː w ᵻ ɡ aʊ m iː t h ɚ w ɛ n z d eɪ æ t ð ə t ɹ eɪ n s t eɪ ʃ ə n"
# given_transcript = "p l iː z k ɔː l s t ɛ l ə æ s k ɚ t ə b ɹ ɪ ŋ ð iː z θ ɪ ŋ z w ɪ ð ɚ f ɹ ʌ m ð ə s t oʊ s ɪ k s p uː n z ʌ v f ɹ ɛ ʃ n uː p iː z f aɪ v θ ɪ k s l æ b z ʌ v b l uː tʃ iː z æ n d m eɪ b i ɐ s n æ k f ɔːɹ h ɚ b ɹ ʌ ð ɚ b oʊ b w iː ɔː l s oʊ n iː d ɐ s m ɔː l p l æ s t ɪ k s n eɪ k æ n d ɐ b ɛ ɡ t ɔːɹ f ɹ oʊ ɡ f ɚ ð ə k ɪ d z ʃ iː k æ n s k uː p ð z θ ɪ ŋ z ɪ n t ə ʊ θ ɹ iː ɹ ɛ d b æ ɡ z æ n d w iː w əl ɡ oʊ m iː ɾ ɚ w ɛ n z d eɪ æ t ð ə t ɹ eɪ n s t eɪ ʃ ə n"
# given_transcript = "p l iː s k oʊ l s t ɛ l ɚ ɑː s k ɚ t ə b ɹ ɪ ŋ ð iː z θ ɪ ŋ z w ɪ v h ɚ f ɹ ʌ m ð ə s t oʊ s ɪ k s p uː n z ʌ v f ɹ ɪ s n oʊ p iː s f aɪ v θ ɪ k s l æ p s ʌ v b l uː tʃ iː z æ n d m eɪ b i ɐ s n æ k f ɔːɹ h ɜː b ɹ ʌ ð ɚ ɚ b ɹ æ ð ɚ b ɔ w iː ɔː l s oʊ n iː d ɐ s m oʊ l p l æ s t ɪ k s n eɪ k æ n d ɐ b ɪ ɡ t uː ɐ f ɹ ɔ k f ɔːɹ ð ə k ɪ d z ʃ iː k æ n s k uː p ð iː z θ ɪ ŋ z ɪ n t ʊ f ɹ iː ɹ ɪ t b æ k s æ n w iː w ɪ l ɡ oʊ m iː ɾ ɐ w ɪ n s t eɪ æ t ə d ɹ eɪ n s t eɪ ʃ ə n"

given_transcript = "p l ɪ s k ɔː l s ɛ t ɚ æ s k h ɜː t ə b ɹ ɪ ŋ ð ɪ s t ɪ ŋ ɡ ə z w ɪ z h ɜː f ɹ ʌ m ð ə s t ɔːɹ s ɪ k s p ʊ n ʌ v f ɹ ɛ ʃ s n oʊ p iː z f aɪ v t ɪ k s s l eɪ p ʌ v b l uː tʃ iː z æ n d m eɪ b i ɪ s n ʌ k f ɔːɹ h ɚ b ɹ aɪ z ɚ b ɑː p w iː ɔː l s oʊ n iː d ɐ s m ɔː l p l æ s t ɪ k s n æ k æ n d b ɪ ɡ t ɔː f ɹ ɑː ɡ f ɔːɹ h ɚ k ɪ d s ð ə s k æ n ɪ s k ɪ v ð ɪ s t ɪ ŋ ɡ z ɪ n t ə θ ɹ iː ɹ iː d b æ k s æ n d w iː w ɪ l ɡ oʊ m ɪ ɾ ɚ w ɛ n s d eɪ æ t ð ə t ɹ eɪ n s t eɪ ʃ ə n"
# American G2P transcription
american_g2p = "pliz kɔl ˈstɛlə. æsk hər tə brɪŋ ðiz θɪŋz wɪð hər frəm ðə stɔr: sɪks spunz əv frɛʃ snoʊ piz, faɪv θɪk slæbz əv blu ʧiz, ənd ˈmeɪbi ə snæk fər hər ˈbrʌðər bɑb. wi ˈɔlsoʊ nid ə smɔl ˈplæstɪk sneɪk ənd ə bɪɡ tɔɪ frɑɡ fər ðə kɪdz. ʃi kən skup ðiz θɪŋz ˈɪntə θri rɛd bæɡz, ənd wi wɪl ɡoʊ mit hər ˈwɛnzˌdeɪ ət ðə treɪn ˈsteɪʃən"

# British G2P transcription
british_g2p = "pliːz kɔːl ˈstɛlə. ɑːsk hə tə brɪŋ ðiːz θɪŋz wɪð hə frəm ðə stɔː: sɪks spuːnz əv frɛʃ snəʊ piːz, faɪv θɪk slæbz əv bluː ʧiːz, ənd ˈmeɪbi ə snæk fə hə ˈbrʌðə bɒb. wi ˈɔːlsəʊ niːd ə smɔːl ˈplæstɪk sneɪk ənd ə bɪɡ tɔɪ frɒɡ fə ðə kɪdz. ʃi kən skuːp ðiːz θɪŋz ˈɪntə θriː rɛd bæɡz, ənd wi wɪl ɡəʊ miːt hə ˈwɛnzdeɪ ət ðə treɪn ˈsteɪʃᵊn"

# strip spaces from all
given_transcript = given_transcript.strip().replace(" ", "")
american_g2p = american_g2p.strip().replace(" ", "")
british_g2p = british_g2p.strip().replace(" ", "")

# Compute Levenshtein Distance
dist_american = per(given_transcript, american_g2p)
dist_british = per(given_transcript, british_g2p)

print(
    "Levenshtein Distance between given transcript and American G2P transcription: ",
    dist_american,
)
print(
    "Levenshtein Distance between given transcript and British G2P transcription: ",
    dist_british,
)

feature_error_rate = fer(given_transcript, american_g2p)
print(
    "Feature Error Rate between given transcript and American G2P transcription: ",
    feature_error_rate,
)

feature_error_rate = fer(given_transcript, british_g2p)
print(
    "Feature Error Rate between given transcript and British G2P transcription: ",
    feature_error_rate,
)
