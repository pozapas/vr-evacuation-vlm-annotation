{COMMON}

TIME FORMAT - CONDITION B (frame sequence)
This item is a sequence of {N} still frames sampled at exactly 1 frame per second, in chronological
order. Frame 0 is the first image, frame {NMAX} the last. There is NO AUDIO: judge the alarm from
the visual change alone (the room flashes red).

Report every time as a FRAME INDEX (an integer 0-{NMAX}, or a decimal between two frames if you can
interpolate). Do not report values outside 0-{NMAX}.
