# Golden set labelling notes (data/golden/golden.jsonl)

220 rows from `candidates.jsonl`, hand-labelled by reading each row's `customer_text` plus `full_thread`
(no model involved). Fields: intent (agent/intents.py TAXONOMY key), should_escalate, escalation_reason, hard, note.

## Rules applied
- **Label what the customer needs, judged from the whole thread; the brand's reply is a tiebreak, not ground truth.**
  If the brand answered one of two issues, that issue wins (`29`: wifi playback + can't upgrade -> playback_issue).
- **Quote-tweets.** 9 rows have `customer_text` = a Spotify promo/status tweet (Hulu-student, "Get Spotify Free",
  "We're investigating payment issues") with the real question in the next customer turn. Labelled the real
  question the brand answered, marked `hard` (a classifier seeing only `customer_text` cannot get these).
- **Premium paid but not active / no skips after upgrade = billing_subscription**, not login or playback (`42, 103, 119, 130, 137`).
- **Delete account, Facebook link, hacked, "someone is using my account" = login_account** even when money is
  mentioned (`116` close-account-blocked-by-Premium, `136` can't-login + wants to quit).
- **Country not available ("launch in India/South Africa/UAE") = content_missing** (region availability), not feature_feedback (`3, 49, 164, 219`).
- **App on a TV / console = device_connect** (taxonomy lists TV/console there); app on the phone/computer it runs on = playback_issue (`70` Xbox, `113, 175` smart TV vs `140, 192`).
- **Bluetooth headphones choppy = device_connect** (`38`); ads playing over music / stuck ad / short ad-free window = playback_issue (`68, 97, 143, 145`); ad *frequency* complaint = feature_feedback (`59`).
- **Algorithm complaints (shuffle, Discover Weekly, Release Radar, curated playlists) = feature_feedback** even when phrased as a bug (`43, 47, 186`).
- **Editorial playlist inclusion request = feature_feedback** (`132`); "add / bring back song X" = content_missing.
- **Artist/label business (profile merge, credits, artist playlists) = other + unclear_or_offtopic**: consumer support can't act, Artist Services can (`50, 87, 146, 163, 191`).
- **Non-English = other + unclear_or_offtopic** even when the content is a real issue (`131` Filipino payment, `162` Spanish).
- **Praise / positive chatter with no request = feature_feedback, auto** (`17, 19, 69, 123, 151, 158`).

### Escalation
- Escalate: any login_account or billing_subscription that needs the account looked at; `other` (unclear, image-only,
  one-word, off-topic, "check DM"); explicit repeat contact ("still waiting", "for months", "already contacted support");
  PII in the tweet; anger only when it carries a churn/legal threat or is the whole message (`10, 92, 107, 181`).
- Auto: troubleshooting, how-tos, content availability, feedback acknowledgement, **and plan/offer questions the brand
  answered publicly with a link** (`23, 27, 44, 80, 134, 147, 174, 180, 184` - billing intent does not force escalation).
- Profanity/shouting without a threat stays auto (`15` "search bar is complete shit", `186`).
- Hacked account is escalated even where the brand only posted a help link (`207, 208`): a human should verify.
- Reason precedence when several apply: pii > repeat_contact > anger_or_legal > account_access / billing_or_refund > unclear_or_offtopic.
  billing_subscription rows use billing_or_refund unless the ask is purely account plumbing (family invite `141`, provider expiry `26`).

## Counts
| intent | n | escalated |
|---|---|---|
| feature_feedback | 57 | 3 |
| billing_subscription | 39 | 28 |
| content_missing | 33 | 3 |
| playback_issue | 31 | 1 |
| other | 22 | 22 |
| login_account | 21 | 20 |
| device_connect | 9 | 1 |
| downloads_library | 8 | 3 |

Escalate rate: **81/220 = 36.8%**. Reasons: billing_or_refund 26, account_access 22, unclear_or_offtopic 22,
repeat_contact 6, anger_or_legal 4, pii 1. `hard` = 67/220 (30%); the 20 regex "hard" candidates plus the
quote-tweet rows account for most of it.

## Hardest cases
| id | why |
|---|---|
| 189825 (`34`) | customer_text is the Hulu promo; thread has 354 turns from many users; labelled the military-discount question the brand answered |
| 1767684 (`105`) | customer_text is a joke ("Roses are red..."); brand replied to a different user's iPhone X question -> other |
| 100177 (`83`) | customer_text is Spotify's own "All clear!" status tweet; real complaint is "your website is down" |
| 565409 (`136`) | can't log in + paying + wants to quit + emails + IBAN in tweet: login_account, escalated for pii |
| 2070844 (`10`) | "Wtf ... on my PAID Spotify, I'll give my $$ to Apple": content answer is standard, but churn threat -> escalate |
| 555492 (`130`) | "upgraded to premium but can't skip": reads like playback, is really Premium not active -> billing |
| 2139022 (`116`) | can't close account because Premium is active: account deletion (login_account) vs billing |
| 1751394 (`172`) | web player redirects to login page: playback (web player) vs login_account |
| 1875152 (`70`) | Xbox app won't start: app-on-own-device (playback) vs console (device_connect) |
| 508845 (`132`) | "add Lemon to Today's Top Hits": content request vs curation feedback |
| 539199 (`131`) | Filipino tweet about a failed Premium payment: other by the language rule, billing by content |
| 2733037 (`209`) | "still having issues for over a month": repeat contact, but brand just kept troubleshooting |

## Brand replies that were bad (for the report)
DM-for-account on things that need no account: `47` (duplicate on Discover Weekly), `106` (settings suggestion),
`161` (content removal), `218` (add a song), `9` (joke tweet). `12` reply read as sarcastic and escalated the customer.
`207/208` answered hacked accounts with a public link only.
