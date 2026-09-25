# Neutral-prompt ablation, analysis plan fixed before any call

Written 2026-09-23, before the first call. Commit this file's hash with the prompt hash. Nothing
below may be changed after the results are seen. Any later analysis is reported as exploratory.

## Why
The v2 task description told every model that the participant "walks to the exit door, and leaves
the room, which ends the waiting-room scene", and required an `exit_reached` timestamp. A reviewer
can read the 73.8% exit-claim rate as instruction following. This ablation removes the assertion and
the obligation and changes nothing else.

## Manipulation
- Prompt `evactime/prompts/common_task_v3_neutral.md`, sha256
  93e8c6f9edb6b219aebfcac67799984663bbb1773bdb2d2b18ce605279fe5f17.
- The only differences from v2 (sha256 c1e3d22c...a2e40) are these three. The scenario sentence
  asserting the exit is replaced by "The recording continues for some time after the alarm."
  `exit_reached` becomes a conditional event, to be reported only if the recording shows it and
  otherwise null. The "best estimate rather than refusing" instruction is restricted to the two
  required events.
- The condition texts, frozen assets, temperatures (rep 0 at 0.0, reps 1 and 2 at 0.4), media
  resolution, and the 29 participants are identical to v2.

## Cells (3 repeats x 29 participants each, so 87 calls per cell and 435 in all)
- Frames (B): GPT-5.2, Claude Opus 4.8 and Qwen3-VL 235B, via OpenRouter. These span the v2 range
  at 13.8%, 79.3% and 96.6%.
- Gemini 3 Flash via the native API, video with audio (A) and frames (B). This tests whether the
  higher exit-claim rate under video (v2, 81.6% for A against 58.6% for B) survives without the
  assertion.

## Primary outcome
The narrative exit claim, scored by `evactime/transit_classifier.asserts_transit` exactly as in v2,
on the usable records from `vlm_records.load_records` rules (at least 20 narrative words, same JSON
recovery). The unit is the call. The comparison is paired on model x participant x condition x
repeat against the v2 record.

## Analyses
1. Exit-claim rate per cell under v3 against v2, with the paired difference and a 95% interval from
   a participant-clustered bootstrap (5,000 draws), and an exact McNemar test on the discordant
   pairs pooled over cells.
2. Pooled rate across all v3 calls, with the same participant-clustered interval.
3. Gemini 3 Flash, the video minus frames difference under v3, compared with the same difference
   under v2 (difference in differences, participant-clustered bootstrap).
4. Field behavior. The share of v3 calls that fill `exit_reached` with a non-null object, and the
   agreement between a filled field and a narrative exit claim.

## How each outcome will be reported
- If the v3 rate stays well above zero, the models narrate an exit the recording does not show even
  when nothing tells them to, and the v2 result is not an artifact of the prompt.
- If the v3 rate falls sharply, the task description drives the claim, and the paper reports that
  models defer to a stated expectation over the visual evidence, with the size of that effect
  measured here directly.
- Anything in between is reported as measured, as the share of the v2 rate that the assertion
  accounts for (1 minus the v3 rate divided by the v2 rate, per cell).
No threshold for "well above zero" or "sharply" is used in the paper. The paper reports the rates,
the paired differences and the intervals.

## Exclusions
Calls lost to provider errors are retried up to three times and otherwise reported as missing,
never imputed. No participant is excluded beyond the v2 exclusions.

## Amendment 1, 2026-09-23, before any Gemini call under v3
The native Gemini API refused every call (quota of 0 requests per minute for the project in region
us-south1), so the two Gemini 3 Flash cells cannot run as planned. They run instead through
OpenRouter (`google/gemini-3-flash-preview`), which accepts video with audio and frames. A test
call showed that OpenRouter encodes the 75 s clip in about 4,950 video and 1,875 audio tokens,
which is not the native "medium" media resolution used for v2. Comparing v3 on OpenRouter against
v2 on the native API would therefore change the route and the prompt together.

To keep the prompt the only manipulated variable, BOTH prompts are run on OpenRouter for Gemini 3
Flash, in video with audio (A) and frames (B), 29 participants x 3 repeats, 348 calls. For these two
cells the primary comparison (analyses 1 and 3) is v3 on OpenRouter against v2 on OpenRouter,
paired on participant, condition and repeat. The native v2 calls are used only in an added
secondary check, the agreement between v2 on OpenRouter and v2 on the native API, which measures
whether the route alone moves the exit-claim rate. The three OpenRouter frame cells (GPT-5.2,
Opus 4.8, Qwen) are unaffected.

## Correction to Amendment 1, 2026-09-23, after the run
Amendment 1 said OpenRouter encodes video at a different resolution from the native "medium"
setting. The logged token counts show otherwise. Video with audio cost a median of 7,546 prompt
tokens natively and 7,548 through OpenRouter, so the video encoding matches. Frames cost a median of
40,292 tokens natively and 83,269 through OpenRouter, so the route difference lies in the frames. The
paired design of Amendment 1 is unaffected, because both prompts ran on the same route.

## Additional analysis, not in the plan (reported as such)
Narratives naming a hallway or corridor together with an exit claim, counted after the run was
seen: 17 of 435 under v2 and 82 of 435 under v3 on the same paired calls.

## Amendment 2, 2026-09-23, before any call it describes
The v2 calls for GPT-5.2, Claude Opus 4.8 and Qwen3-VL 235B date from 2026-07-23, two months before
their v3 calls, and OpenRouter served them from partly different backends (Opus through Anthropic in
July and through Claude Platform on AWS today, Qwen partly through Novita in July and only through
Alibaba today). Date and backend are therefore confounded with the prompt in these three cells. To
remove the confound, v2 is rerun today for the three cells, 29 participants x 3 repeats, with
OpenRouter pinned to the backend each model used for its v3 calls and fallbacks disabled. The primary
comparison for these cells becomes v3 against same-day v2. The July v2 calls enter only an added
drift check, same-day v2 against July v2, which measures how far date and backend alone move the
rate. Analyses 1 to 4 are otherwise unchanged.

Recomputed after Amendment 2 and after the classifier extension recorded in
docs/CLASSIFIER_REVIEW_NEUTRAL.md: 16 of 435 under v2 (same-day) and 86 of 435 under v3. The earlier
counts above (17 and 82) used the July v2 calls and the unextended classifier.

## Amendment 3, 2026-09-23, before any call it describes
A new Gemini key restored the native interface. Gemini 3 Flash is now also run natively, with both
descriptions on the same day, in video with audio (A) and frames (B), 29 participants x 3 repeats,
348 calls, media resolution medium and temperatures as in v2. This restores the Gemini cells of the
original plan on their original route. Analyses: (i) v3 against same-day v2 on the native route,
paired as in analysis 1; (ii) video minus frames on the native route under each description, as in
analysis 3; (iii) same-day comparison of the two routes under each description. The 435-call pooled
result of analyses 1 and 2 is unchanged; the native cells are reported as a second-route replication.
