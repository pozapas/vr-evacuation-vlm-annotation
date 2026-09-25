You are annotating a first-person (head-mounted display) recording from a virtual-reality
fire-evacuation experiment. A participant is seated in a clinic waiting room. During the recording a
FIRE ALARM activates: a red visual alarm makes the room flash, and in some recordings an audible
alarm also sounds. The participant then stands, walks to the exit door, and leaves the room, which
ends the waiting-room scene.

Identify these THREE events. Each occurs exactly once and each MUST be reported:

  alarm_onset    the first instant the fire alarm becomes perceptible (the room first flashes red,
                 or the alarm sound starts)
  movement_start the first instant the participant begins to TRAVEL through the room in response to
                 the alarm - the viewpoint starts translating. Turning the head or looking around
                 while stationary does NOT count.
  exit_reached   the instant the participant passes through the exit door and the waiting-room
                 scene ends

Report as many of these OPTIONAL events as you can also see: door_interaction, npc_stands,
scene_change, hesitation, backtrack, look_at_exit_sign, look_at_alarm, look_at_npc.

Also produce:
  - phases:     a contiguous segmentation into baseline / detection / recognition / decision /
                movement / wayfinding / egress
  - wayfinding: approach_exit, orient_to_sign, backtrack, mill, search, open_door
  - social:     look_at_npc, follow_npc, wait_for_npc
  - narrative:  3-5 plain sentences describing what happens
  - timeline:   the salient moments an analyst should review
  - flags:      occlusion, ambiguous_movement, no_audio_available, overall_uncertain

Be conservative: give your single best estimate rather than refusing, but set a low confidence and
raise the relevant flag when you are unsure. Return STRICT JSON conforming to the schema. No prose
outside the JSON.
