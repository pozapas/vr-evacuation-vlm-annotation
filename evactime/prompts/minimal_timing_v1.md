You are annotating a first-person (head-mounted display) recording from a virtual-reality
fire-evacuation experiment. A participant is seated in a clinic waiting room. During the recording a
FIRE ALARM activates (the room flashes red; in some recordings an alarm also sounds). The
participant then stands, walks to the exit door, and leaves the room.

Report the time of exactly these three events. Each occurs exactly once and each MUST be reported.

  alarm_onset     first instant the fire alarm becomes perceptible
  movement_start  first instant the participant begins to TRAVEL through the room (the viewpoint
                  starts translating). Head turns while stationary do NOT count.
  exit_reached    instant the participant passes through the exit door and the scene ends

Give your single best estimate rather than refusing; use confidence to express uncertainty.

{TIME_FORMAT}

Return STRICT JSON only, no prose:
{"events":[{"type":"alarm_onset","t":<number>,"confidence":<0-1>},
           {"type":"movement_start","t":<number>,"confidence":<0-1>},
           {"type":"exit_reached","t":<number>,"confidence":<0-1>}],
 "narrative":"<one sentence>"}
