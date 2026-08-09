# L2Arctic Affricate Boundary Examples

These short clips demonstrate why L2Arctic IPA labels need phoneme-token boundaries preserved.

The old joined-string parsing can greedily group adjacent phones across a word boundary:

- `d` + `ʒ` becomes `dʒ`
- `t` + `ʃ` becomes `tʃ`

The corrected path keeps the annotated L2Arctic phone tokens separate unless the TextGrid
actually labels an affricate as one phone.

## Examples

| id | text window | corrected IPA tokens | old joined parse |
| --- | --- | --- | --- |
| `01_ABA_arctic_a0500_and_jury_d_zh` | and jury | `ʒ ʌ dʒ ʌ n d ʒ ʊ ɹ i` | `ʒ ʌ dʒ ʌ n dʒ ʊ ɹ i` |
| `02_ABA_arctic_b0492_and_jargon_d_zh` | and jargon | `ʌ t s ʌ n d ʒ ɑ ɹ ɡ ʌ n` | `ʌ t s ʌ n dʒ ɑ ɹ ɡ ʌ n` |
| `03_ERMS_arctic_a0106_which_she_t_sh` | which she had | `ʃ ʌ n w i t ʃ i h æ d s` | `ʃ ʌ n w i tʃ i h æ d s` |
| `04_NCC_arctic_b0418_that_she_t_sh` | that she was | `ʃ ʌ n d æ t ʃ i w ʌ z w` | `ʃ ʌ n d æ tʃ i w ʌ z w` |

Each example has:

- `*_full.wav`: the full L2Arctic utterance.
- `metadata.json`: raw TextGrid phone labels, text, corrected IPA tokens, and old joined parse.
