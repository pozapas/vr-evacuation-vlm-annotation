# Hand review of the transit classifier on neutral-prompt narratives, 2026-09-23

Why. The classifier's self-tests and its validation against the human audit use narratives written
under the original task description. Narratives written under the neutral description phrase the
ending differently, and the 119 calls that dropped the exit claim carry the ablation result.

Sample. 40 of the 119 calls that dropped the claim (random_state=7), plus the 6 neutral-prompt
narratives naming a hallway or corridor that the classifier did not flag. One narrative is in both
sets, so 45 unique narratives were read.

Finding. 7 of the 45 assert that the participant left and were missed, all in one construction:
"passes (through) into an adjoining/adjacent room/space/corridor" (5, all Claude Opus 4.8) and "the
recording shows the participant in the hallway" (1, Gemini 3 Flash), plus one Opus "passes through
into a further corridor/room". The remaining 38 were correct: accurate descriptions of the reset,
a door opening while the participant stays, an attempt that fails, or a recording that ends first.

Fix. Two patterns were added to TRANSIT_PATTERNS with six new self-tests (43 in all, 0 failures).
After the fix the classifier agrees with the hand review on all 45. On the original-description
corpus the fix flags 11 more narratives, all read and all genuine exit claims ("exit into the
adjacent corridor", "pass through into an adjacent room"), which moves the corpus rate from 73.8%
to 74.7%. Against the human audit labels it still flags no narrative the coders judged correct,
and recall is 0.875.
