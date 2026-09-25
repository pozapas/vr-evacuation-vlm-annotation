You are annotating a first-person (head-mounted display) recording from a virtual-reality
fire-evacuation experiment. A participant is seated in a clinic waiting room. During the recording a
FIRE ALARM activates: a red visual alarm makes the room flash, and in some recordings an audible
alarm also sounds. The participant then stands, walks to the exit door, and leaves the room, which
ends the waiting-room scene.

THREE REQUIRED EVENTS - each occurs exactly once and each MUST be reported:

  alarm_onset     the first moment the fire alarm becomes perceptible (the room first flashes red,
                  or the alarm sound starts)
  movement_start  the first moment the participant begins to TRAVEL through the room in response to
                  the alarm - the viewpoint starts translating. Turning the head or looking around
                  while stationary does NOT count.
  exit_reached    the moment the participant passes through the exit door and the waiting-room
                  scene ends

Give your single best estimate rather than refusing; express doubt through `confidence`, not by
omitting the field.

Return STRICT JSON matching EXACTLY this shape. Use {UNIT} for every time value.

{
  "events": {
    "alarm_onset":    {"{TKEY}": <number>, "confidence": <0-1>, "evidence": "<short clause>"},
    "movement_start": {"{TKEY}": <number>, "confidence": <0-1>, "evidence": "<short clause>"},
    "exit_reached":   {"{TKEY}": <number>, "confidence": <0-1>, "evidence": "<short clause>"}
  },
  "optional_events": [
    {"type": "<door_interaction|npc_stands|scene_change|hesitation|backtrack|look_at_exit_sign|look_at_alarm|look_at_npc>",
     "{TKEY}": <number>, "confidence": <0-1>}
  ],
  "phases": [
    {"phase": "<baseline|detection|recognition|decision|movement|wayfinding|egress>",
     "start": <number>, "end": <number>}
  ],
  "wayfinding": [
    {"action": "<approach_exit|orient_to_sign|backtrack|mill|search|open_door>",
     "start": <number>, "end": <number>}
  ],
  "social": [
    {"action": "<look_at_npc|follow_npc|wait_for_npc>", "start": <number>, "end": <number>}
  ],
  "narrative": "<3-5 plain sentences describing what happens>",
  "timeline": [{"{TKEY}": <number>, "label": "<short label for a salient moment>"}],
  "flags": {"occlusion": <bool>, "ambiguous_movement": <bool>,
            "no_audio_available": <bool>, "overall_uncertain": <bool>}
}

No prose outside the JSON. Do not add or rename keys.
